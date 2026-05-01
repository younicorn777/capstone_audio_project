import os
import csv
import numpy as np
import librosa


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")
MA_DIR = os.path.join(BASE_DIR, "results", "moving_average")
KF_DIR = os.path.join(BASE_DIR, "results", "kalman_best")

EVAL_DIR = os.path.join(BASE_DIR, "results", "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(EVAL_DIR, "filter_evaluation_results.csv")

SR = 16000


# =========================
# 2. 오디오 로드 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def align_length(a, b):
    """
    두 오디오 길이를 짧은 쪽에 맞춤
    """
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


# =========================
# 3. 평가 지표 함수
# =========================

def mse(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return np.mean((clean - estimate) ** 2)


def mae(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return np.mean(np.abs(clean - estimate))


def snr_db(clean, estimate):
    """
    clean을 기준 신호로 보고,
    clean - estimate를 noise/error로 보아 SNR 계산
    """
    clean, estimate = align_length(clean, estimate)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - estimate) ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def snr_improvement(clean, noisy, estimate):
    """
    SNR Improvement = 필터 적용 후 SNR - 필터 적용 전 SNR
    """
    before = snr_db(clean, noisy)
    after = snr_db(clean, estimate)

    return after - before


# =========================
# 4. 파일 이름 매칭 함수
# =========================

def get_clean_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav -> clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")

    # parts = ["noisy", "clean01", "air"]
    clean_number = parts[1].replace("clean", "")

    return f"clean_{clean_number}.wav"


def get_ma_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav -> ma_clean01_air.wav
    """
    return noisy_file.replace("noisy_", "ma_")


def get_kf_best_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav -> kf_best_clean01_air.wav
    """
    return noisy_file.replace("noisy_", "kf_best_")


# =========================
# 5. 평가 실행
# =========================

def main():
    print("필터 성능 평가 시작")
    print("-" * 50)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    results = []

    for noisy_file in noisy_files:
        clean_file = get_clean_file_from_noisy(noisy_file)
        ma_file = get_ma_file_from_noisy(noisy_file)
        kf_file = get_kf_best_file_from_noisy(noisy_file)

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

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)

        methods = {
            "Noisy": noisy,
        }

        if os.path.exists(ma_path):
            methods["Moving Average"] = load_audio(ma_path)
        else:
            print(f"[경고] Moving Average 파일 없음: {ma_path}")

        if os.path.exists(kf_path):
            methods["Kalman Best"] = load_audio(kf_path)
        else:
            print(f"[경고] Kalman Best 파일 없음: {kf_path}")

        for method_name, estimate in methods.items():
            row = {
                "file": noisy_file.replace(".wav", ""),
                "clean_file": clean_file,
                "method": method_name,
                "MSE": mse(clean, estimate),
                "MAE": mae(clean, estimate),
                "SNR_dB": snr_db(clean, estimate),
                "SNR_Improvement_dB": (
                    0.0 if method_name == "Noisy"
                    else snr_improvement(clean, noisy, estimate)
                ),
            }

            results.append(row)

            print(
                f"{row['file']} | {method_name} | "
                f"MSE={row['MSE']:.6f}, "
                f"MAE={row['MAE']:.6f}, "
                f"SNR={row['SNR_dB']:.2f} dB, "
                f"Improvement={row['SNR_Improvement_dB']:.2f} dB"
            )

    # CSV 저장
    fieldnames = [
        "file",
        "clean_file",
        "method",
        "MSE",
        "MAE",
        "SNR_dB",
        "SNR_Improvement_dB",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 50)
    print(f"평가 결과 저장 완료: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()