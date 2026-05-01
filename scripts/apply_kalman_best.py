import os
import numpy as np
import librosa
import soundfile as sf


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

NOISY_DIR = os.path.join(BASE_DIR, "noisy")
RESULT_DIR = os.path.join(BASE_DIR, "results", "kalman_best")

os.makedirs(RESULT_DIR, exist_ok=True)

SR = 16000


# =========================
# 2. 파일별 Best Q/R 설정
# =========================

BEST_QR = {
    "noisy_clean01_air.wav": (1e-4, 1e-2),
    "noisy_clean01_engine.wav": (1e-6, 1e-2),
    "noisy_clean01_dog.wav": (1e-5, 1e-2),

    "noisy_clean02_air.wav": (1e-4, 1e-1),
    "noisy_clean02_engine.wav": (1e-6, 1e-2),
    "noisy_clean02_dog.wav": (1e-5, 1e-2),

    "noisy_clean03_air.wav": (1e-4, 1e-1),
    "noisy_clean03_engine.wav": (1e-6, 1e-2),
    "noisy_clean03_dog.wav": (1e-5, 1e-2),
}


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


# =========================
# 4. 2차 상태 칼만필터
# =========================

def kalman_filter_audio(noisy_audio, q_scale, r_scale):
    """
    2차 상태 칼만필터 적용

    상태 x = [s, ds]^T
    s  : 실제 음성 진폭
    ds : 음성 진폭 변화율

    관측값 z = noisy audio amplitude
    """

    n = len(noisy_audio)

    A = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    H = np.array([[1.0, 0.0]])

    Q = q_scale * np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    R = np.array([[r_scale]])

    x = np.array([
        [noisy_audio[0]],
        [0.0]
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
# 5. Best Q/R 적용
# =========================

def main():
    print("Best Q/R 기반 Kalman Filter 적용 시작")
    print("-" * 50)

    for noisy_file, (q_scale, r_scale) in BEST_QR.items():
        input_path = os.path.join(NOISY_DIR, noisy_file)

        if not os.path.exists(input_path):
            print(f"[경고] 파일 없음: {input_path}")
            continue

        noisy_audio = load_audio(input_path)

        filtered_audio = kalman_filter_audio(
            noisy_audio=noisy_audio,
            q_scale=q_scale,
            r_scale=r_scale,
        )

        output_name = noisy_file.replace("noisy_", "kf_best_")
        output_path = os.path.join(RESULT_DIR, output_name)

        sf.write(output_path, filtered_audio, SR)

        print(f"[저장 완료] {output_name} | Q={q_scale}, R={r_scale}")

    print("-" * 50)
    print("Best Q/R 기반 Kalman Filter 적용 완료!")


if __name__ == "__main__":
    main()