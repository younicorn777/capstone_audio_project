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
OUTPUT_DIR = os.path.join(BASE_DIR, "noisy_vad")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 16000
TARGET_SNR_DB = 5

NOISE_ONLY_FRONT_SEC = 1.0
NOISE_ONLY_BACK_SEC = 0.5


# =========================
# 2. 파일 목록
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
# 3. 유틸 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val


def match_noise_length(noise, target_length):
    """
    noise 길이를 target_length에 맞춤
    - 짧으면 반복
    - 길면 자름
    """
    if len(noise) < target_length:
        repeat_count = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_count)

    return noise[:target_length]


def scale_noise_to_snr(clean, noise, snr_db):
    """
    clean 구간 기준으로 noise 크기를 SNR에 맞게 조절
    """
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        raise ValueError("Noise power is zero. 다른 noise 파일을 사용하세요.")

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    return noise * scale


def prevent_clipping(audio):
    """
    -1~1 범위를 넘을 때만 스케일 조정
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    return audio


# =========================
# 4. Noisy Voice 생성
# =========================

def main():
    print("Noise-only 구간 포함 Noisy Voice 생성 시작")
    print(f"Target SNR: {TARGET_SNR_DB} dB")
    print(f"Front noise-only: {NOISE_ONLY_FRONT_SEC} sec")
    print(f"Back noise-only: {NOISE_ONLY_BACK_SEC} sec")
    print("-" * 60)

    front_len = int(NOISE_ONLY_FRONT_SEC * SR)
    back_len = int(NOISE_ONLY_BACK_SEC * SR)

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

            # 전체 필요한 noise 길이
            total_len = front_len + len(clean) + back_len
            noise_full = match_noise_length(noise, total_len)

            # 구간 분리
            noise_front = noise_full[:front_len]
            noise_for_clean = noise_full[front_len:front_len + len(clean)]
            noise_back = noise_full[front_len + len(clean):]

            # clean 구간 기준으로 SNR 맞춤
            scaled_noise_for_clean = scale_noise_to_snr(
                clean=clean,
                noise=noise_for_clean,
                snr_db=TARGET_SNR_DB,
            )

            # noise-only 구간도 같은 scale을 적용하기 위해 scale 다시 계산
            # scale_noise_to_snr 함수 내부 계산을 재사용하지 않기 위해 직접 계산
            clean_power = np.mean(clean ** 2)
            noise_power = np.mean(noise_for_clean ** 2)
            target_noise_power = clean_power / (10 ** (TARGET_SNR_DB / 10))
            scale = np.sqrt(target_noise_power / noise_power)

            scaled_noise_front = noise_front * scale
            scaled_noise_back = noise_back * scale

            # 중간 구간: clean + noise
            noisy_middle = clean + scaled_noise_for_clean

            # 최종 연결
            noisy_with_noise_only = np.concatenate([
                scaled_noise_front,
                noisy_middle,
                scaled_noise_back,
            ])

            noisy_with_noise_only = prevent_clipping(noisy_with_noise_only)

            output_name = f"noisy_vad_clean{clean_id}_{noise_label}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            sf.write(output_path, noisy_with_noise_only, SR)

            duration_sec = len(noisy_with_noise_only) / SR
            print(f"[저장 완료] {output_name} / 길이: {duration_sec:.2f} sec")

    print("-" * 60)
    print("Noise-only 구간 포함 Noisy Voice 생성 완료!")


if __name__ == "__main__":
    main()
