import os
import csv
import numpy as np
import librosa
import soundfile as sf


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy_vad")
VAD_CSV = os.path.join(BASE_DIR, "results", "vad", "vad_segments.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results", "spectral_subtraction_grid_search")
BEST_AUDIO_DIR = os.path.join(RESULT_DIR, "best_audio")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BEST_AUDIO_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "spectral_subtraction_grid_search_results.csv")
BEST_CSV = os.path.join(RESULT_DIR, "spectral_subtraction_best_results.csv")

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256


# =========================
# 2. alpha / beta 후보군
# =========================

ALPHA_CANDIDATES = [
    0.5,
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
]

BETA_CANDIDATES = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
]


# =========================
# 3. 오디오 유틸 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def prevent_clipping(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    return audio


def align_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


# =========================
# 4. 평가 지표
# =========================

def mse(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return np.mean((clean - estimate) ** 2)


def mae(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return np.mean(np.abs(clean - estimate))


def snr_db(clean, estimate):
    clean, estimate = align_length(clean, estimate)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - estimate) ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def snr_improvement(clean, noisy, estimate):
    before = snr_db(clean, noisy)
    after = snr_db(clean, estimate)
    return after - before


# =========================
# 5. 파일 이름 매칭
# =========================

def get_clean_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")

    # ["noisy", "vad", "clean01", "air"]
    clean_number = parts[2].replace("clean", "")

    return f"clean_{clean_number}.wav"


def get_short_name(noisy_file):
    """
    noisy_vad_clean01_air.wav -> clean01_air
    """
    return noisy_file.replace("noisy_vad_", "").replace(".wav", "")


# =========================
# 6. VAD CSV 읽기
# =========================

def load_vad_segments(vad_csv_path):
    vad_dict = {}

    with open(vad_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            file_name = row["file"]
            segment_type = row["segment_type"]
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])

            if file_name not in vad_dict:
                vad_dict[file_name] = {
                    "speech": [],
                    "non_speech": [],
                }

            vad_dict[file_name][segment_type].append({
                "start": start_sec,
                "end": end_sec,
            })

    return vad_dict


# =========================
# 7. noise profile 추정
# =========================

def extract_noise_audio(audio, non_speech_segments, sr=SR):
    noise_parts = []

    for seg in non_speech_segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        # 너무 짧은 non-speech 구간은 제외
        if end_sample - start_sample < int(0.1 * sr):
            continue

        noise_parts.append(audio[start_sample:end_sample])

    if len(noise_parts) == 0:
        return None

    return np.concatenate(noise_parts)


def estimate_noise_magnitude(noise_audio):
    noise_stft = librosa.stft(
        noise_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    noise_mag = np.abs(noise_stft)

    # frequency bin별 평균 noise magnitude
    noise_profile = np.mean(noise_mag, axis=1, keepdims=True)

    return noise_profile


# =========================
# 8. Spectral Subtraction
# =========================

def spectral_subtraction(noisy_audio, noise_profile, alpha, beta):
    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    noisy_mag = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)

    subtracted_mag = noisy_mag - alpha * noise_profile

    # spectral floor 적용
    floor = beta * noisy_mag
    enhanced_mag = np.maximum(subtracted_mag, floor)

    enhanced_stft = enhanced_mag * np.exp(1j * noisy_phase)

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=HOP_LENGTH,
        length=len(noisy_audio)
    )

    enhanced_audio = prevent_clipping(enhanced_audio)

    return enhanced_audio.astype(np.float32)


# =========================
# 9. Grid Search 실행
# =========================

def main():
    print("Spectral Subtraction alpha/beta Grid Search 시작")
    print("-" * 70)

    if not os.path.exists(VAD_CSV):
        print(f"[오류] VAD CSV 파일이 없습니다: {VAD_CSV}")
        return

    vad_dict = load_vad_segments(VAD_CSV)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    all_results = []
    best_results = []

    for noisy_file in noisy_files:
        if noisy_file not in vad_dict:
            print(f"[경고] VAD 정보 없음: {noisy_file}")
            continue

        clean_file = get_clean_file_from_noisy_vad(noisy_file)

        clean_path = os.path.join(CLEAN_DIR, clean_file)
        noisy_path = os.path.join(NOISY_DIR, noisy_file)

        if not os.path.exists(clean_path):
            print(f"[경고] Clean 파일 없음: {clean_path}")
            continue

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)

        # noisy_vad는 앞/뒤 noise-only 구간이 있어서 clean과 길이가 다름
        # 평가할 때는 speech가 들어간 중간 구간과 clean을 비교해야 함
        # 현재 생성 기준: 앞 1초 noise-only, 뒤 0.5초 noise-only
        front_len = int(1.0 * SR)
        clean_len = len(clean)

        noisy_middle = noisy[front_len:front_len + clean_len]

        non_speech_segments = vad_dict[noisy_file]["non_speech"]

        noise_audio = extract_noise_audio(
            audio=noisy,
            non_speech_segments=non_speech_segments,
            sr=SR
        )

        if noise_audio is None:
            print(f"[경고] noise profile 추정 실패: {noisy_file}")
            continue

        noise_profile = estimate_noise_magnitude(noise_audio)

        noisy_mse = mse(clean, noisy_middle)
        noisy_mae = mae(clean, noisy_middle)
        noisy_snr = snr_db(clean, noisy_middle)

        best_row = None
        best_audio = None

        print(f"\n[파일 처리 중] {noisy_file}")

        for alpha in ALPHA_CANDIDATES:
            for beta in BETA_CANDIDATES:
                enhanced_full = spectral_subtraction(
                    noisy_audio=noisy,
                    noise_profile=noise_profile,
                    alpha=alpha,
                    beta=beta
                )

                enhanced_middle = enhanced_full[front_len:front_len + clean_len]

                current_mse = mse(clean, enhanced_middle)
                current_mae = mae(clean, enhanced_middle)
                current_snr = snr_db(clean, enhanced_middle)
                current_improvement = current_snr - noisy_snr

                row = {
                    "file": noisy_file.replace(".wav", ""),
                    "clean_file": clean_file,
                    "alpha": alpha,
                    "beta": beta,
                    "MSE": current_mse,
                    "MAE": current_mae,
                    "SNR_dB": current_snr,
                    "SNR_Improvement_dB": current_improvement,
                    "Noisy_MSE": noisy_mse,
                    "Noisy_MAE": noisy_mae,
                    "Noisy_SNR_dB": noisy_snr,
                }

                all_results.append(row)

                # 1차 기준: MSE 최소
                if best_row is None or current_mse < best_row["MSE"]:
                    best_row = row
                    best_audio = enhanced_full

        best_results.append(best_row)

        output_name = noisy_file.replace("noisy_vad_", "ss_best_")
        output_path = os.path.join(BEST_AUDIO_DIR, output_name)

        sf.write(output_path, best_audio, SR)

        print(
            f"[Best] {noisy_file} | "
            f"alpha={best_row['alpha']}, beta={best_row['beta']}, "
            f"MSE={best_row['MSE']:.6f}, "
            f"MAE={best_row['MAE']:.6f}, "
            f"SNR={best_row['SNR_dB']:.2f} dB, "
            f"Improvement={best_row['SNR_Improvement_dB']:.2f} dB"
        )

    fieldnames = [
        "file",
        "clean_file",
        "alpha",
        "beta",
        "MSE",
        "MAE",
        "SNR_dB",
        "SNR_Improvement_dB",
        "Noisy_MSE",
        "Noisy_MAE",
        "Noisy_SNR_dB",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    with open(BEST_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_results)

    print("\n" + "-" * 70)
    print("Spectral Subtraction Grid Search 완료")
    print(f"전체 결과 CSV: {OUTPUT_CSV}")
    print(f"Best 결과 CSV: {BEST_CSV}")
    print(f"Best audio 폴더: {BEST_AUDIO_DIR}")


if __name__ == "__main__":
    main()