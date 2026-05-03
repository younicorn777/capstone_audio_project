import os
import numpy as np
import librosa
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")
MA_DIR = os.path.join(BASE_DIR, "results", "moving_average")
KF_DIR = os.path.join(BASE_DIR, "results", "kalman_grid_search", "best_audio")

PLOT_DIR = os.path.join(BASE_DIR, "results", "plots", "final_comparison")
os.makedirs(PLOT_DIR, exist_ok=True)

SR = 16000


# =========================
# 2. 오디오 로드 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def make_time_axis(audio, sr=SR):
    return np.arange(len(audio)) / sr


# =========================
# 3. 파일 매칭 함수
# =========================

def get_clean_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav → clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    clean_number = parts[1].replace("clean", "")
    return f"clean_{clean_number}.wav"


def get_ma_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav → ma_clean01_air.wav
    """
    return noisy_file.replace("noisy_", "ma_")


def get_kf_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav → kf_grid_best_clean01_air.wav
    """
    return noisy_file.replace("noisy_", "kf_grid_best_")


# =========================
# 4. 파형 비교 plot 생성
# =========================

def plot_comparison(clean, noisy, ma, kf, title, output_path):
    min_len = min(len(clean), len(noisy), len(ma), len(kf))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    ma = ma[:min_len]
    kf = kf[:min_len]

    t = make_time_axis(clean)

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, clean)
    plt.title("Clean Voice")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)

    plt.subplot(4, 1, 2)
    plt.plot(t, noisy)
    plt.title("Noisy Voice")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)

    plt.subplot(4, 1, 3)
    plt.plot(t, ma)
    plt.title("Moving Average Filter")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)

    plt.subplot(4, 1, 4)
    plt.plot(t, kf)
    plt.title("Kalman Filter")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.ylim(-1, 1)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =========================
# 5. 전체 파일 처리
# =========================

def main():
    print("최종 파형 비교 이미지 생성 시작")
    print("-" * 50)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    if len(noisy_files) == 0:
        print("[오류] noisy 폴더에 wav 파일이 없습니다.")
        return

    for noisy_file in noisy_files:
        clean_file = get_clean_file_from_noisy(noisy_file)
        ma_file = get_ma_file_from_noisy(noisy_file)
        kf_file = get_kf_file_from_noisy(noisy_file)

        clean_path = os.path.join(CLEAN_DIR, clean_file)
        noisy_path = os.path.join(NOISY_DIR, noisy_file)
        ma_path = os.path.join(MA_DIR, ma_file)
        kf_path = os.path.join(KF_DIR, kf_file)

        if not os.path.exists(clean_path):
            print(f"[경고] Clean 파일 없음: {clean_path}")
            continue

        if not os.path.exists(ma_path):
            print(f"[경고] Moving Average 파일 없음: {ma_path}")
            continue

        if not os.path.exists(kf_path):
            print(f"[경고] Kalman 파일 없음: {kf_path}")
            continue

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)
        ma = load_audio(ma_path)
        kf = load_audio(kf_path)

        plot_name = noisy_file.replace(".wav", "_final_waveform.png")
        output_path = os.path.join(PLOT_DIR, plot_name)

        title = noisy_file.replace(".wav", "")
        plot_comparison(clean, noisy, ma, kf, title, output_path)

        print(f"[저장 완료] {plot_name}")

    print("-" * 50)
    print("최종 파형 비교 이미지 생성 완료!")


if __name__ == "__main__":
    main()