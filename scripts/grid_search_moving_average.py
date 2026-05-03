import os
import csv
import numpy as np
import librosa
import soundfile as sf


BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")

RESULT_DIR = os.path.join(BASE_DIR, "results", "moving_average_grid_search")
BEST_AUDIO_DIR = os.path.join(RESULT_DIR, "best_audio")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BEST_AUDIO_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "moving_average_grid_search_results.csv")
BEST_CSV = os.path.join(RESULT_DIR, "moving_average_best_results.csv")

SR = 16000

WINDOW_CANDIDATES = [3, 5, 11, 15, 21, 23, 31, 51, 101]


def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def align_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def moving_average_filter(audio, window_size):
    kernel = np.ones(window_size) / window_size
    pad = window_size // 2
    padded_audio = np.pad(audio, pad_width=pad, mode="reflect")
    filtered = np.convolve(padded_audio, kernel, mode="valid")
    return filtered


def prevent_clipping(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    return audio


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


def get_clean_file_from_noisy(noisy_file):
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    clean_number = parts[1].replace("clean", "")
    return f"clean_{clean_number}.wav"


def main():
    print("Moving Average Window Size Grid Search 시작")
    print("-" * 60)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    all_results = []
    best_results = []

    for noisy_file in noisy_files:
        clean_file = get_clean_file_from_noisy(noisy_file)

        clean_path = os.path.join(CLEAN_DIR, clean_file)
        noisy_path = os.path.join(NOISY_DIR, noisy_file)

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)

        noisy_mse = mse(clean, noisy)
        noisy_mae = mae(clean, noisy)
        noisy_snr = snr_db(clean, noisy)

        best_row = None
        best_audio = None

        print(f"\n[파일 처리 중] {noisy_file}")

        for window_size in WINDOW_CANDIDATES:
            filtered = moving_average_filter(noisy, window_size)
            filtered = prevent_clipping(filtered)

            current_mse = mse(clean, filtered)
            current_mae = mae(clean, filtered)
            current_snr = snr_db(clean, filtered)
            current_improvement = current_snr - noisy_snr

            row = {
                "file": noisy_file.replace(".wav", ""),
                "clean_file": clean_file,
                "window_size": window_size,
                "MSE": current_mse,
                "MAE": current_mae,
                "SNR_dB": current_snr,
                "SNR_Improvement_dB": current_improvement,
                "Noisy_MSE": noisy_mse,
                "Noisy_MAE": noisy_mae,
                "Noisy_SNR_dB": noisy_snr,
            }

            all_results.append(row)

            # MSE 기준으로 best 선택
            if best_row is None or current_mse < best_row["MSE"]:
                best_row = row
                best_audio = filtered

        best_results.append(best_row)

        output_name = noisy_file.replace("noisy_", "ma_best_")
        output_path = os.path.join(BEST_AUDIO_DIR, output_name)
        sf.write(output_path, best_audio, SR)

        print(
            f"[Best] {noisy_file} | "
            f"window={best_row['window_size']}, "
            f"MSE={best_row['MSE']:.6f}, "
            f"MAE={best_row['MAE']:.6f}, "
            f"SNR={best_row['SNR_dB']:.2f} dB, "
            f"Improvement={best_row['SNR_Improvement_dB']:.2f} dB"
        )

    fieldnames = [
        "file",
        "clean_file",
        "window_size",
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

    print("\n" + "-" * 60)
    print("Moving Average Grid Search 완료")
    print(f"전체 결과 CSV: {OUTPUT_CSV}")
    print(f"Best 결과 CSV: {BEST_CSV}")
    print(f"Best audio 폴더: {BEST_AUDIO_DIR}")


if __name__ == "__main__":
    main()
