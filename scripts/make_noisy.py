import os
import numpy as np
import librosa
import soundfile as sf

# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISE_DIR = os.path.join(BASE_DIR, "noise")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")

os.makedirs(NOISY_DIR, exist_ok=True)

SR = 16000
TARGET_SNR_DB = 5


# =========================
# 2. 오디오 유틸 함수
# =========================

def load_audio(path, sr=SR):
    """
    오디오 파일을 mono, 지정 sampling rate로 불러오는 함수
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def normalize_audio(audio):
    """
    오디오 amplitude를 -1 ~ 1 범위로 정규화
    """
    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    return audio / max_val


def match_noise_length(noise, target_length):
    """
    noise 길이를 clean voice 길이에 맞추는 함수

    - noise가 짧으면 반복
    - noise가 길면 자름
    """
    if len(noise) == target_length:
        return noise

    if len(noise) < target_length:
        repeat_count = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_count)

    return noise[:target_length]


def mix_with_snr(clean, noise, snr_db):
    """
    Clean Voice와 Noise를 목표 SNR에 맞게 합성하는 함수

    SNR = 10 * log10(clean_power / noise_power)
    """
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        raise ValueError("Noise power is zero. 다른 noise 파일을 사용하세요.")

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    adjusted_noise = noise * scale
    noisy = clean + adjusted_noise

    # clipping 방지를 위해 최종 결과 정규화
    noisy = normalize_audio(noisy)

    return noisy, adjusted_noise


# =========================
# 3. 파일 목록
# =========================

clean_files = [
    "clean_01.wav",
    "clean_02.wav",
    "clean_03.wav",
]

noise_files = {
    "air": "noise_air_conditioner.wav",
    "engine": "noise_engine_idling.wav",
    "dog": "noise_dog_barking.wav",
}


# =========================
# 4. Noisy Voice 생성
# =========================

def main():
    print("Noisy Voice 생성을 시작합니다.")
    print(f"Target SNR: {TARGET_SNR_DB} dB")
    print("-" * 50)

    for clean_file in clean_files:
        clean_path = os.path.join(CLEAN_DIR, clean_file)

        if not os.path.exists(clean_path):
            print(f"[경고] Clean 파일 없음: {clean_path}")
            continue

        clean = load_audio(clean_path)
        clean = normalize_audio(clean)

        clean_id = clean_file.replace("clean_", "").replace(".wav", "")

        for noise_label, noise_file in noise_files.items():
            noise_path = os.path.join(NOISE_DIR, noise_file)

            if not os.path.exists(noise_path):
                print(f"[경고] Noise 파일 없음: {noise_path}")
                continue

            noise = load_audio(noise_path)
            noise = normalize_audio(noise)

            noise_matched = match_noise_length(noise, len(clean))

            noisy, adjusted_noise = mix_with_snr(
                clean=clean,
                noise=noise_matched,
                snr_db=TARGET_SNR_DB,
            )

            output_name = f"noisy_clean{clean_id}_{noise_label}.wav"
            output_path = os.path.join(NOISY_DIR, output_name)

            sf.write(output_path, noisy, SR)

            duration_sec = len(noisy) / SR
            print(f"[저장 완료] {output_name} / 길이: {duration_sec:.2f} sec")

    print("-" * 50)
    print("Noisy Voice 생성 완료!")


if __name__ == "__main__":
    main()