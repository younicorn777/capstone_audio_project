## 시연용 음성 파일 생성

import os
import numpy as np
import librosa
import soundfile as sf

BASE_DIR = "capstone_week9_dataset"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "kalman_grid_search",
    "best_audio"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "demo_audio",
    "kalman_best_normalized"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 16000
TARGET_PEAK = 0.90


def peak_normalize(audio, target_peak=TARGET_PEAK):
    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    normalized = audio / max_val * target_peak
    return normalized


def main():
    print("Kalman Best 시연용 음량 보정 시작")
    print("-" * 50)

    wav_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".wav")
    ])

    if len(wav_files) == 0:
        print("[오류] 입력 폴더에 wav 파일이 없습니다.")
        return

    for file_name in wav_files:
        input_path = os.path.join(INPUT_DIR, file_name)

        audio, _ = librosa.load(input_path, sr=SR, mono=True)

        normalized_audio = peak_normalize(audio)

        output_name = file_name.replace("kf_grid_best_", "demo_kf_best_")
        output_path = os.path.join(OUTPUT_DIR, output_name)

        sf.write(output_path, normalized_audio, SR)

        print(f"[저장 완료] {output_name}")

    print("-" * 50)
    print("시연용 음량 보정 완료!")


if __name__ == "__main__":
    main()