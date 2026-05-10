import os
import csv
import numpy as np
import librosa
import soundfile as sf
from pystoi import stoi


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy_vad")
VAD_CSV = os.path.join(BASE_DIR, "results", "vad", "vad_segments.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results", "wiener_filter_grid_search")
BEST_AUDIO_DIR = os.path.join(RESULT_DIR, "best_audio")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BEST_AUDIO_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "wiener_filter_grid_search_results.csv")
BEST_CSV = os.path.join(RESULT_DIR, "wiener_filter_best_results.csv")

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256

FRONT_NOISE_ONLY_SEC = 1.0


# =========================
# 2. Wiener Filter 후보 파라미터
# =========================

NOISE_SCALE_CANDIDATES = [
    0.5,
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
]

GAIN_FLOOR_CANDIDATES = [
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


def zero_mean(signal):
    return signal - np.mean(signal)


# =========================
# 4. 평가 지표
# =========================

def mse(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return float(np.mean((clean - estimate) ** 2))


def mae(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return float(np.mean(np.abs(clean - estimate)))


def snr_db(clean, estimate):
    clean, estimate = align_length(clean, estimate)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - estimate) ** 2)

    if noise_power == 0:
        return float("inf")

    return float(10 * np.log10(signal_power / noise_power))


def si_snr_db(clean, estimate, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio 계산
    """
    clean, estimate = align_length(clean, estimate)

    clean = zero_mean(clean)
    estimate = zero_mean(estimate)

    clean_energy = np.sum(clean ** 2) + eps

    target = (np.sum(estimate * clean) / clean_energy) * clean
    noise = estimate - target

    target_power = np.sum(target ** 2)
    noise_power = np.sum(noise ** 2) + eps

    return float(10 * np.log10((target_power + eps) / noise_power))


def stoi_score(clean, estimate):
    """
    STOI 계산
    - 0~1 범위
    - 1에 가까울수록 음성 명료도가 좋음
    """
    clean, estimate = align_length(clean, estimate)

    try:
        score = stoi(clean, estimate, SR, extended=False)
        return float(score)
    except Exception as e:
        print(f"[경고] STOI 계산 실패: {e}")
        return np.nan


# =========================
# 5. 파일 이름 매칭
# =========================

def get_clean_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")

    clean_number = parts[2].replace("clean", "")

    return f"clean_{clean_number}.wav"


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
# 7. Noise Power Profile 추정
# =========================

def extract_noise_audio(audio, non_speech_segments, sr=SR):
    noise_parts = []

    for seg in non_speech_segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        if end_sample - start_sample < int(0.1 * sr):
            continue

        noise_parts.append(audio[start_sample:end_sample])

    if len(noise_parts) == 0:
        return None

    return np.concatenate(noise_parts)


def estimate_noise_power(noise_audio):
    noise_stft = librosa.stft(
        noise_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    noise_power = np.abs(noise_stft) ** 2

    noise_power_profile = np.mean(noise_power, axis=1, keepdims=True)

    return noise_power_profile


# =========================
# 8. Wiener Filter
# =========================

def wiener_filter(noisy_audio, noise_power_profile, noise_scale, gain_floor):
    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    noisy_mag = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)
    noisy_power = noisy_mag ** 2

    scaled_noise_power = noise_scale * noise_power_profile

    estimated_speech_power = noisy_power - scaled_noise_power
    estimated_speech_power = np.maximum(estimated_speech_power, 0.0)

    gain = estimated_speech_power / (
        estimated_speech_power + scaled_noise_power + 1e-12
    )

    gain = np.maximum(gain, gain_floor)

    enhanced_mag = gain * noisy_mag
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
    print("Wiener Filter noise_scale / gain_floor Grid Search 시작")
    print("Best 기준: SI-SNR Improvement 최대")
    print("-" * 70)

    if not os.path.exists(VAD_CSV):
        print(f"[오류] VAD CSV 파일이 없습니다: {VAD_CSV}")
        return

    vad_dict = load_vad_segments(VAD_CSV)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    if len(noisy_files) == 0:
        print(f"[오류] 입력 폴더에 wav 파일이 없습니다: {NOISY_DIR}")
        return

    all_results = []
    best_results = []

    front_len = int(FRONT_NOISE_ONLY_SEC * SR)

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

        noise_power_profile = estimate_noise_power(noise_audio)

        noisy_mse = mse(clean, noisy_middle)
        noisy_mae = mae(clean, noisy_middle)
        noisy_snr = snr_db(clean, noisy_middle)
        noisy_si_snr = si_snr_db(clean, noisy_middle)
        noisy_stoi = stoi_score(clean, noisy_middle)

        best_row = None
        best_audio = None

        print(f"\n[파일 처리 중] {noisy_file}")

        for noise_scale in NOISE_SCALE_CANDIDATES:
            for gain_floor in GAIN_FLOOR_CANDIDATES:
                enhanced_full = wiener_filter(
                    noisy_audio=noisy,
                    noise_power_profile=noise_power_profile,
                    noise_scale=noise_scale,
                    gain_floor=gain_floor
                )

                enhanced_middle = enhanced_full[front_len:front_len + clean_len]

                current_mse = mse(clean, enhanced_middle)
                current_mae = mae(clean, enhanced_middle)
                current_snr = snr_db(clean, enhanced_middle)
                current_snr_improvement = current_snr - noisy_snr

                current_si_snr = si_snr_db(clean, enhanced_middle)
                current_si_snr_improvement = current_si_snr - noisy_si_snr

                current_stoi = stoi_score(clean, enhanced_middle)

                row = {
                    "file": noisy_file.replace(".wav", ""),
                    "clean_file": clean_file,
                    "noise_scale": noise_scale,
                    "gain_floor": gain_floor,

                    "MSE": current_mse,
                    "MAE": current_mae,
                    "SNR_dB": current_snr,
                    "SNR_Improvement_dB": current_snr_improvement,

                    "SI_SNR_dB": current_si_snr,
                    "SI_SNR_Improvement_dB": current_si_snr_improvement,
                    "STOI": current_stoi,

                    "Noisy_MSE": noisy_mse,
                    "Noisy_MAE": noisy_mae,
                    "Noisy_SNR_dB": noisy_snr,
                    "Noisy_SI_SNR_dB": noisy_si_snr,
                    "Noisy_STOI": noisy_stoi,
                }

                all_results.append(row)

                # Best 선택 기준:
                # 1순위: SI-SNR Improvement 최대
                # 2순위: STOI 최대
                # 3순위: MSE 최소
                if best_row is None:
                    best_row = row
                    best_audio = enhanced_full
                else:
                    better = False

                    if row["SI_SNR_Improvement_dB"] > best_row["SI_SNR_Improvement_dB"]:
                        better = True
                    elif row["SI_SNR_Improvement_dB"] == best_row["SI_SNR_Improvement_dB"]:
                        if row["STOI"] > best_row["STOI"]:
                            better = True
                        elif row["STOI"] == best_row["STOI"] and row["MSE"] < best_row["MSE"]:
                            better = True

                    if better:
                        best_row = row
                        best_audio = enhanced_full

        best_results.append(best_row)

        output_name = noisy_file.replace("noisy_vad_", "wf_best_")
        output_path = os.path.join(BEST_AUDIO_DIR, output_name)

        sf.write(output_path, best_audio, SR)

        print(
            f"[Best] {noisy_file} | "
            f"noise_scale={best_row['noise_scale']}, "
            f"gain_floor={best_row['gain_floor']}, "
            f"SI-SNRi={best_row['SI_SNR_Improvement_dB']:.2f} dB, "
            f"STOI={best_row['STOI']:.4f}, "
            f"SNRi={best_row['SNR_Improvement_dB']:.2f} dB, "
            f"MSE={best_row['MSE']:.6f}"
        )

    fieldnames = [
        "file",
        "clean_file",
        "noise_scale",
        "gain_floor",

        "MSE",
        "MAE",
        "SNR_dB",
        "SNR_Improvement_dB",

        "SI_SNR_dB",
        "SI_SNR_Improvement_dB",
        "STOI",

        "Noisy_MSE",
        "Noisy_MAE",
        "Noisy_SNR_dB",
        "Noisy_SI_SNR_dB",
        "Noisy_STOI",
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
    print("Wiener Filter Grid Search 완료")
    print(f"전체 결과 CSV: {OUTPUT_CSV}")
    print(f"Best 결과 CSV: {BEST_CSV}")
    print(f"Best audio 폴더: {BEST_AUDIO_DIR}")


if __name__ == "__main__":
    main()