import os
import csv
import numpy as np
import librosa
import soundfile as sf


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

NOISY_DIR = os.path.join(BASE_DIR, "noisy_vad")
VAD_CSV = os.path.join(BASE_DIR, "results", "vad", "vad_segments.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results", "spectral_subtraction")
os.makedirs(RESULT_DIR, exist_ok=True)

SR = 16000

N_FFT = 1024
HOP_LENGTH = 256

# Spectral Subtraction 파라미터 (1.5/0.05)
ALPHA = 1.3 # noise spectrum을 얼마나 강하게 뺄지
BETA = 0.02 # 너무 많이 제거되어 0이 되는 것을 방지하는 spectral floor


# =========================
# 2. 오디오 유틸 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def prevent_clipping(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    return audio


# =========================
# 3. VAD CSV 읽기
# =========================

def load_vad_segments(vad_csv_path):
    """
    vad_segments.csv를 읽어서 파일별 segment 정보를 dictionary로 정리
    """
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
# 4. noise profile 추정
# =========================

def extract_noise_audio(audio, non_speech_segments, sr=SR):
    """
    non-speech 구간에서 noise-only audio를 추출
    """
    noise_parts = []

    for seg in non_speech_segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        # 너무 짧은 구간은 제외
        if end_sample - start_sample < int(0.1 * sr):
            continue

        noise_parts.append(audio[start_sample:end_sample])

    if len(noise_parts) == 0:
        return None

    noise_audio = np.concatenate(noise_parts)
    return noise_audio


def estimate_noise_magnitude(noise_audio):
    """
    noise-only audio에서 평균 noise magnitude spectrum 추정
    """
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
# 5. Spectral Subtraction
# =========================

def spectral_subtraction(noisy_audio, noise_profile):
    """
    Noisy spectrum에서 noise spectrum을 빼는 방식
    """
    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    noisy_mag = np.abs(noisy_stft)
    noisy_phase = np.angle(noisy_stft)

    # Spectral subtraction
    subtracted_mag = noisy_mag - ALPHA * noise_profile

    # spectral floor 적용
    floor = BETA * noisy_mag
    enhanced_mag = np.maximum(subtracted_mag, floor)

    # 원래 phase와 결합
    enhanced_stft = enhanced_mag * np.exp(1j * noisy_phase)

    # time domain 복원
    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=HOP_LENGTH,
        length=len(noisy_audio)
    )

    enhanced_audio = prevent_clipping(enhanced_audio)

    return enhanced_audio.astype(np.float32)


# =========================
# 6. 전체 파일 처리
# =========================

def main():
    print("Spectral Subtraction 적용 시작")
    print(f"ALPHA: {ALPHA}")
    print(f"BETA: {BETA}")
    print("-" * 60)

    if not os.path.exists(VAD_CSV):
        print(f"[오류] VAD CSV 파일이 없습니다: {VAD_CSV}")
        return

    vad_dict = load_vad_segments(VAD_CSV)

    wav_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    if len(wav_files) == 0:
        print(f"[오류] 입력 폴더에 wav 파일이 없습니다: {NOISY_DIR}")
        return

    for wav_file in wav_files:
        input_path = os.path.join(NOISY_DIR, wav_file)

        if wav_file not in vad_dict:
            print(f"[경고] VAD 정보 없음: {wav_file}")
            continue

        audio = load_audio(input_path)

        non_speech_segments = vad_dict[wav_file]["non_speech"]

        noise_audio = extract_noise_audio(
            audio=audio,
            non_speech_segments=non_speech_segments,
            sr=SR
        )

        if noise_audio is None:
            print(f"[경고] noise profile 추정 실패: {wav_file}")
            continue

        noise_profile = estimate_noise_magnitude(noise_audio)

        enhanced_audio = spectral_subtraction(
            noisy_audio=audio,
            noise_profile=noise_profile
        )

        output_name = wav_file.replace("noisy_vad_", "ss_")
        output_path = os.path.join(RESULT_DIR, output_name)

        sf.write(output_path, enhanced_audio, SR)

        print(f"[저장 완료] {output_name}")

    print("-" * 60)
    print("Spectral Subtraction 적용 완료!")
    print(f"결과 저장 위치: {RESULT_DIR}")


if __name__ == "__main__":
    main()