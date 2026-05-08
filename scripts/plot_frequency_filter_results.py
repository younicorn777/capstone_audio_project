import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy_vad")

SS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "spectral_subtraction_grid_search",
    "best_audio"
)

WF_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wiener_filter_grid_search",
    "best_audio"
)

PLOT_DIR = os.path.join(BASE_DIR, "results", "plots", "frequency_filter_comparison")
WAVEFORM_DIR = os.path.join(PLOT_DIR, "waveform")
SPECTROGRAM_DIR = os.path.join(PLOT_DIR, "spectrogram")

os.makedirs(WAVEFORM_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)

SR = 16000
N_FFT = 1024
HOP_LENGTH = 256

FRONT_NOISE_ONLY_SEC = 1.0


# =========================
# 2. 오디오 로드 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def make_time_axis(audio, sr=SR):
    return np.arange(len(audio)) / sr


# =========================
# 3. 파일 이름 매칭 함수
# =========================

def get_clean_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    clean_number = parts[2].replace("clean", "")
    return f"clean_{clean_number}.wav"


def get_ss_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> ss_best_clean01_air.wav
    """
    return noisy_file.replace("noisy_vad_", "ss_best_")


def get_wf_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> wf_best_clean01_air.wav
    """
    return noisy_file.replace("noisy_vad_", "wf_best_")


def extract_middle_region(audio, clean_len):
    """
    앞 1초 noise-only 구간을 제외하고,
    clean voice와 대응되는 중간 구간만 추출
    """
    front_len = int(FRONT_NOISE_ONLY_SEC * SR)
    return audio[front_len:front_len + clean_len]


# =========================
# 4. Waveform Plot
# =========================

def plot_waveform_comparison(clean, noisy, ss, wf, title, output_path):
    min_len = min(len(clean), len(noisy), len(ss), len(wf))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    ss = ss[:min_len]
    wf = wf[:min_len]

    t = make_time_axis(clean)

    max_amp = max(
        np.max(np.abs(clean)),
        np.max(np.abs(noisy)),
        np.max(np.abs(ss)),
        np.max(np.abs(wf))
    )

    ylim = max_amp * 1.1
    if ylim == 0:
        ylim = 1.0

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(t, clean, linewidth=0.8)
    axes[0].set_title("Clean Voice")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(-ylim, ylim)

    axes[1].plot(t, noisy, linewidth=0.8)
    axes[1].set_title("Noisy Voice")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_ylim(-ylim, ylim)

    axes[2].plot(t, ss, linewidth=0.8)
    axes[2].set_title("Spectral Subtraction")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_ylim(-ylim, ylim)

    axes[3].plot(t, wf, linewidth=0.8)
    axes[3].set_title("Wiener Filter")
    axes[3].set_xlabel("Time (sec)")
    axes[3].set_ylabel("Amplitude")
    axes[3].set_ylim(-ylim, ylim)

    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(hspace=0.45, top=0.92)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================
# 5. Spectrogram Plot
# =========================

def compute_spectrogram_db(audio):
    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectrogram_db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    return spectrogram_db


def plot_single_spectrogram(audio, title, ax):
    spectrogram_db = compute_spectrogram_db(audio)

    img = librosa.display.specshow(
        spectrogram_db,
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    ax.set_title(title, pad=10)
    ax.set_ylabel("Frequency (Hz)")

    return img


def plot_spectrogram_comparison(clean, noisy, ss, wf, title, output_path):
    min_len = min(len(clean), len(noisy), len(ss), len(wf))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    ss = ss[:min_len]
    wf = wf[:min_len]

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

    img = plot_single_spectrogram(clean, "Clean Voice", axes[0])
    plot_single_spectrogram(noisy, "Noisy Voice", axes[1])
    plot_single_spectrogram(ss, "Spectral Subtraction", axes[2])
    plot_single_spectrogram(wf, "Wiener Filter", axes[3])

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("Time (sec)", labelpad=12)

    fig.suptitle(title, fontsize=16, y=0.98)

    # 수정 코드
    cbar_ax = fig.add_axes([0.92, 0.07, 0.02, 0.86])  # [left, bottom, width, height]
    fig.colorbar(
        img,
        cax=cbar_ax,
        format="%+2.0f dB",
        label="Amplitude (dB)"
    )

    fig.subplots_adjust(
        top=0.93,
        bottom=0.07,
        left=0.08,
        right=0.90,
        hspace=0.55
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================
# 6. 전체 실행
# =========================

def main():
    print("주파수 영역 필터 비교 이미지 생성 시작")
    print("-" * 60)

    # 발표용 대표 샘플만 우선 생성
    target_files = [
        "noisy_vad_clean01_air.wav",
        "noisy_vad_clean01_engine.wav",
        "noisy_vad_clean01_dog.wav",
    ]

    for noisy_file in target_files:
        clean_file = get_clean_file_from_noisy_vad(noisy_file)
        ss_file = get_ss_file_from_noisy_vad(noisy_file)
        wf_file = get_wf_file_from_noisy_vad(noisy_file)

        clean_path = os.path.join(CLEAN_DIR, clean_file)
        noisy_path = os.path.join(NOISY_DIR, noisy_file)
        ss_path = os.path.join(SS_DIR, ss_file)
        wf_path = os.path.join(WF_DIR, wf_file)

        if not os.path.exists(clean_path):
            print(f"[경고] Clean 파일 없음: {clean_path}")
            continue

        if not os.path.exists(noisy_path):
            print(f"[경고] Noisy 파일 없음: {noisy_path}")
            continue

        if not os.path.exists(ss_path):
            print(f"[경고] Spectral Subtraction 파일 없음: {ss_path}")
            continue

        if not os.path.exists(wf_path):
            print(f"[경고] Wiener Filter 파일 없음: {wf_path}")
            continue

        clean = load_audio(clean_path)
        noisy_full = load_audio(noisy_path)
        ss_full = load_audio(ss_path)
        wf_full = load_audio(wf_path)

        clean_len = len(clean)

        # 평가와 동일하게 중간 clean+noise 구간만 비교
        noisy = extract_middle_region(noisy_full, clean_len)
        ss = extract_middle_region(ss_full, clean_len)
        wf = extract_middle_region(wf_full, clean_len)

        base_name = noisy_file.replace(".wav", "")

        waveform_path = os.path.join(
            WAVEFORM_DIR,
            base_name + "_waveform.png"
        )

        spectrogram_path = os.path.join(
            SPECTROGRAM_DIR,
            base_name + "_spectrogram.png"
        )

        plot_title = base_name

        plot_waveform_comparison(
            clean=clean,
            noisy=noisy,
            ss=ss,
            wf=wf,
            title=plot_title,
            output_path=waveform_path
        )

        plot_spectrogram_comparison(
            clean=clean,
            noisy=noisy,
            ss=ss,
            wf=wf,
            title=plot_title,
            output_path=spectrogram_path
        )

        print(f"[저장 완료] {base_name}")

    print("-" * 60)
    print("주파수 영역 필터 비교 이미지 생성 완료")
    print(f"Waveform 저장 위치: {WAVEFORM_DIR}")
    print(f"Spectrogram 저장 위치: {SPECTROGRAM_DIR}")


if __name__ == "__main__":
    main()
