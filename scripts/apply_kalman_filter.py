import os
import numpy as np
import librosa
import soundfile as sf


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

NOISY_DIR = os.path.join(BASE_DIR, "noisy")
RESULT_DIR = os.path.join(BASE_DIR, "results", "kalman")

os.makedirs(RESULT_DIR, exist_ok=True)

SR = 16000

# Kalman Filter 파라미터
# Q: 상태 모델 불확실성
# R: 측정값 불확실성

'''

# air: 지속적이고, 안정적인 배경 소음
# air의 best: Q_SCALE = 1e-4 / R_SCALE = 1e-2

# dog: 순간적으로 튀는 소음 => 관측값을 덜 믿어야 함(C,D,E)
# dog의 best: Q_SCALE = 1e-6 / R_SCALE = 1e-2

# engine: 지속적이지만, 음성과 주파수 대역이 일부 겹칠 수 있음 (C,D,E)
# engine의 best: Q_SCALE = 1e-5 / R_SCALE = 1e-1

# A. 현재 기본값
Q_SCALE = 1e-5
R_SCALE = 1e-2

# B. 관측값을 조금 더 믿음
Q_SCALE = 1e-5
R_SCALE = 1e-3

# C. 관측값을 덜 믿음
Q_SCALE = 1e-5
R_SCALE = 1e-1

# D. 더 부드럽게 추정
Q_SCALE = 1e-6
R_SCALE = 1e-2

# E. 강한 smoothing
Q_SCALE = 1e-6
R_SCALE = 1e-1

# F. 음성 변화 더 따라감
Q_SCALE = 1e-4
R_SCALE = 1e-3
'''

Q_SCALE = 1e-5
R_SCALE = 1e-1


# =========================
# 2. 오디오 유틸 함수
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def prevent_clipping(audio):
    """
    값이 -1~1 범위를 넘을 때만 스케일 조정
    """
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    return audio


# =========================
# 3. 2차 상태 칼만필터
# =========================

def kalman_filter_audio(noisy_audio, q_scale=Q_SCALE, r_scale=R_SCALE):
    """
    2차 상태 칼만필터를 오디오 신호에 적용

    상태 x = [s, ds]^T
    - s  : 실제 음성 진폭
    - ds : 음성 진폭 변화율

    관측값 z = noisy audio amplitude
    """

    n = len(noisy_audio)

    # 상태전이 행렬
    A = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])

    # 관측 행렬
    H = np.array([[1.0, 0.0]])

    # 프로세스 노이즈 공분산
    Q = q_scale * np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # 관측 노이즈 공분산
    R = np.array([[r_scale]])

    # 초기 상태
    x = np.array([
        [noisy_audio[0]],
        [0.0]
    ])

    # 초기 오차 공분산
    P = np.eye(2)

    filtered = np.zeros(n)

    for k in range(n):
        z = np.array([[noisy_audio[k]]])

        # -------------------------
        # 1) Prediction
        # -------------------------
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # -------------------------
        # 2) Update
        # -------------------------
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(2) - K @ H) @ P_pred

        # 추정된 음성 진폭 저장
        filtered[k] = x[0, 0]

    filtered = prevent_clipping(filtered)

    return filtered


# =========================
# 4. 전체 noisy 파일 처리
# =========================

def main():
    print("2차 상태 칼만필터 적용 시작")
    print(f"Q_SCALE: {Q_SCALE}")
    print(f"R_SCALE: {R_SCALE}")
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

        noisy_audio = load_audio(input_path)
        filtered_audio = kalman_filter_audio(noisy_audio)

        output_name = noisy_file.replace("noisy_", "kf_")
        output_path = os.path.join(RESULT_DIR, output_name)

        sf.write(output_path, filtered_audio, SR)

        print(f"[저장 완료] {output_name}")

    print("-" * 50)
    print("2차 상태 칼만필터 적용 완료!")


if __name__ == "__main__":
    main()