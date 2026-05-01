import os
import csv
import numpy as np
import librosa
import soundfile as sf


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")

RESULT_DIR = os.path.join(BASE_DIR, "results", "kalman_grid_search")
BEST_AUDIO_DIR = os.path.join(RESULT_DIR, "best_audio")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BEST_AUDIO_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "kalman_qr_grid_search_results.csv")
BEST_CSV = os.path.join(RESULT_DIR, "kalman_qr_best_results.csv")

SR = 16000


# =========================
# 2. Q/R 후보군
# =========================

Q_CANDIDATES = [
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
]

R_CANDIDATES = [
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
]


# =========================
# 3. 오디오 유틸 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def prevent_clipping(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    return audio


def align_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


# =========================
# 4. 평가 지표
# =========================

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


# =========================
# 5. 파일 이름 매칭
# =========================

def get_clean_file_from_noisy(noisy_file):
    """
    noisy_clean01_air.wav -> clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    clean_number = parts[1].replace("clean", "")
    return f"clean_{clean_number}.wav"


# =========================
# 6. 2차 상태 칼만필터
# =========================

def kalman_filter_audio(noisy_audio, q_scale, r_scale):
    """
    상태 x = [s, ds]^T
    s  : 실제 음성 진폭
    ds : 음성 진폭 변화율

    관측값 z = noisy audio amplitude
    """

    n = len(noisy_audio)

    A = np.array([
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    H = np.array([[1.0, 0.0]])

    Q = q_scale * np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    R = np.array([[r_scale]])

    x = np.array([
        [noisy_audio[0]],
        [0.0],
    ])

    P = np.eye(2)

    filtered = np.zeros(n)

    for k in range(n):
        z = np.array([[noisy_audio[k]]])

        # Prediction
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred

        filtered[k] = x[0, 0]

    filtered = prevent_clipping(filtered)

    return filtered


# =========================
# 7. Grid Search 실행
# =========================

def main():
    print("Kalman Q/R Grid Search 시작")
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

        if not os.path.exists(clean_path):
            print(f"[경고] Clean 파일 없음: {clean_path}")
            continue

        if not os.path.exists(noisy_path):
            print(f"[경고] Noisy 파일 없음: {noisy_path}")
            continue

        clean = load_audio(clean_path)
        noisy = load_audio(noisy_path)

        # Noisy 기준 성능
        noisy_mse = mse(clean, noisy)
        noisy_mae = mae(clean, noisy)
        noisy_snr = snr_db(clean, noisy)

        best_row = None
        best_audio = None

        print(f"\n[파일 처리 중] {noisy_file}")

        for q in Q_CANDIDATES:
            for r in R_CANDIDATES:
                filtered = kalman_filter_audio(noisy, q, r)

                current_mse = mse(clean, filtered)
                current_mae = mae(clean, filtered)
                current_snr = snr_db(clean, filtered)
                current_improvement = current_snr - noisy_snr

                row = {
                    "file": noisy_file.replace(".wav", ""),
                    "clean_file": clean_file,
                    "Q": q,
                    "R": r,
                    "MSE": current_mse,
                    "MAE": current_mae,
                    "SNR_dB": current_snr,
                    "SNR_Improvement_dB": current_improvement,
                    "Noisy_MSE": noisy_mse,
                    "Noisy_MAE": noisy_mae,
                    "Noisy_SNR_dB": noisy_snr,
                }

                all_results.append(row)

                # MSE가 가장 낮은 조합을 best로 선택
                if best_row is None or current_mse < best_row["MSE"]:
                    best_row = row
                    best_audio = filtered

        best_results.append(best_row)

        output_name = noisy_file.replace("noisy_", "kf_grid_best_")
        output_path = os.path.join(BEST_AUDIO_DIR, output_name)
        sf.write(output_path, best_audio, SR)

        print(
            f"[Best] {noisy_file} | "
            f"Q={best_row['Q']}, R={best_row['R']}, "
            f"MSE={best_row['MSE']:.6f}, "
            f"MAE={best_row['MAE']:.6f}, "
            f"SNR={best_row['SNR_dB']:.2f} dB, "
            f"Improvement={best_row['SNR_Improvement_dB']:.2f} dB"
        )

    # 전체 grid search 결과 저장
    fieldnames = [
        "file",
        "clean_file",
        "Q",
        "R",
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
    print("Grid Search 완료")
    print(f"전체 결과 CSV: {OUTPUT_CSV}")
    print(f"Best 결과 CSV: {BEST_CSV}")
    print(f"Best Kalman audio 폴더: {BEST_AUDIO_DIR}")


if __name__ == "__main__":
    main()