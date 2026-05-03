import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")
MA_DIR = os.path.join(BASE_DIR, "results", "moving_average")
KF_DIR = os.path.join(BASE_DIR, "results", "kalman_grid_search", "best_audio")

PLOT_DIR = os.path.join(BASE_DIR, "results", "plots", "final_spectrogram")
os.makedirs(PLOT_DIR, exist_ok=True)

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256


def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def get_clean_file_from_noisy(noisy_file):
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    clean_number = parts[1].replace("clean", "")
    return f"clean_{clean_number}.wav"


def get_ma_file_from_noisy(noisy_file):
    return noisy_file.replace("noisy_", "ma_")


def get_kf_file_from_noisy(noisy_file):
    return noisy_file.replace("noisy_", "kf_grid_best_")


def plot_spectrogram(audio, sr, title, ax):
    spectrogram = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    img = librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    return img


def plot_comparison(clean, noisy, ma, kf, title, output_path):
    min_len = min(len(clean), len(noisy), len(ma), len(kf))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    ma = ma[:min_len]
    kf = kf[:min_len]

    # figsize의 높이를 조금 더 여유 있게 조정 (12 -> 14)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    img = plot_spectrogram(clean, SR, "Clean Voice", axes[0])
    plot_spectrogram(noisy, SR, "Noisy Voice", axes[1])
    plot_spectrogram(ma, SR, "Moving Average Filter", axes[2])
    plot_spectrogram(kf, SR, "Kalman Filter", axes[3])

    axes[3].set_xlabel("Time (sec)")

    # 1. tight_layout을 먼저 호출하여 기본적인 여백을 잡습니다.
    # rect 매개변수는 전체 제목(suptitle)이 들어갈 공간을 상단에 확보합니다.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 2. hspace를 조절하여 서브플롯 간의 세로 간격을 명시적으로 벌립니다.
    # 값은 0.3 ~ 0.5 사이에서 취향에 맞게 조정하세요.
    plt.subplots_adjust(hspace=0.4)

    fig.suptitle(title, fontsize=16)
    
    # 컬러바 위치 조정 (전체 그래프 우측에 배치)
    fig.colorbar(img, ax=axes, format="%+2.0f dB", label="Amplitude (dB)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("최종 스펙트로그램 비교 이미지 생성 시작")
    print("-" * 50)

    # 발표용 대표 샘플만 생성
    target_files = [
        "noisy_clean01_air.wav",
        "noisy_clean01_engine.wav",
        "noisy_clean01_dog.wav",
    ]

    for noisy_file in target_files:
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

        if not os.path.exists(noisy_path):
            print(f"[경고] Noisy 파일 없음: {noisy_path}")
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

        plot_name = noisy_file.replace(".wav", "_final_spectrogram.png")
        output_path = os.path.join(PLOT_DIR, plot_name)

        title = noisy_file.replace(".wav", "")
        plot_comparison(clean, noisy, ma, kf, title, output_path)

        print(f"[저장 완료] {plot_name}")

    print("-" * 50)
    print("최종 스펙트로그램 비교 이미지 생성 완료!")


if __name__ == "__main__":
    main()