import os
import numpy as np
import librosa
import soundfile as sf


# =========================
# 1. 기본 설정
# =========================

# 데이터셋이 들어있는 기본 폴더 이름
BASE_DIR = "capstone_week9_dataset" 

# 잡음이 섞인 오디오 파일들이 들어있는 폴더
NOISY_DIR = os.path.join(BASE_DIR, "noisy")

# 필터 적용 후 결과를 저장할 폴더
RESULT_DIR = os.path.join(BASE_DIR, "results", "moving_average")
os.makedirs(RESULT_DIR, exist_ok=True)

# 오디오 샘플링 레이트 (초당 16000 샘플)
SR = 16000

# 이동평균 window size
# 숫자가 클수록 더 부드러워지지만, 음성이 먹먹해질 수 있음
'''
WINDOW_SIZE = 51  # 음성 보존 좋음, 느이즈 감소 약함
WINDOW_SIZE = 101 # 균형
WINDOW_SIZE = 201 # 노이즈 감소 증가, 음성 약간 먹먹
WINDOW_SIZE = 401 # 강한 smoothing, 음성 왜곡 가능
'''
WINDOW_SIZE = 101


# =========================
# 2. 유틸 함수
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


def moving_average_filter(audio, window_size):
    """
    이동평균 필터 적용 함수

    audio: 입력 오디오 신호
    window_size: 평균을 낼 샘플 개수
    """
    if window_size < 1:
        raise ValueError("window_size는 1 이상이어야 합니다.")

    if window_size % 2 == 0:
        raise ValueError("window_size는 홀수로 설정하는 것을 권장합니다.")

    kernel = np.ones(window_size) / window_size

    # mode='same'을 사용하면 입력과 출력 길이가 동일함
    filtered = np.convolve(audio, kernel, mode="same")

    return filtered


# =========================
# 3. 이동평균 필터 적용
# =========================

def main():
    print("이동평균 필터 적용 시작")
    print(f"Window size: {WINDOW_SIZE}")
    print("-" * 50)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    if len(noisy_files) == 0:
        print("[오류] noisy 폴더에 wav 파일이 없습니다.")
        return

    for noisy_file in noisy_files:
        input_path = os.path.join(NOISY_DIR, noisy_file)

        # 오디오 불러오기 -> 필터 적용 -> 정규화 -> 저장
        noisy_audio = load_audio(input_path)
        filtered_audio = moving_average_filter(noisy_audio, WINDOW_SIZE)
        filtered_audio = normalize_audio(filtered_audio)

        output_name = noisy_file.replace("noisy_", "ma_")
        output_path = os.path.join(RESULT_DIR, output_name)

        sf.write(output_path, filtered_audio, SR)

        print(f"[저장 완료] {output_name}")

    print("-" * 50)
    print("이동평균 필터 적용 완료!")


if __name__ == "__main__":
    main()
