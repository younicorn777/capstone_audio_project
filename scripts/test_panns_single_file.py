import os
import numpy as np
import librosa

from panns_inference import AudioTagging, labels


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

# 테스트할 wav 파일 하나 선택
# verified dataset에 있는 파일 중 하나로 테스트
TEST_WAV = os.path.join(
    BASE_DIR,
    "external_noise_dataset_verified",
    "dog",
    "dog_001.wav"
)

SR = 32000  # PANNs는 보통 32 kHz 입력을 사용


# =========================
# 2. 오디오 로드
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)

    # PANNs 입력은 batch 형태 필요: [batch, time]
    audio = audio[None, :]

    return audio


# =========================
# 3. main
# =========================

def main():
    print("PANNs 단일 파일 테스트 시작")
    print("-" * 60)

    if not os.path.exists(TEST_WAV):
        print(f"[오류] 테스트 파일이 없습니다: {TEST_WAV}")
        return

    print(f"테스트 파일: {TEST_WAV}")

    print("PANNs AudioTagging 모델 로딩 중...")
    at = AudioTagging(checkpoint_path=None, device="cpu")

    audio = load_audio(TEST_WAV)

    print("PANNs inference 실행 중...")
    clipwise_output, embedding = at.inference(audio)

    # clipwise_output shape: [batch, class_num]
    # embedding shape: [batch, embedding_dim]
    scores = clipwise_output[0]
    emb = embedding[0]

    print(f"Class score shape: {clipwise_output.shape}")
    print(f"Embedding shape: {embedding.shape}")

    # 상위 10개 클래스 출력
    top_indices = np.argsort(scores)[::-1][:10]

    print("\nTop 10 predicted classes")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"{rank}. {labels[idx]}: {scores[idx]:.4f}")

    print("\nPANNs 단일 파일 테스트 완료")


if __name__ == "__main__":
    main()
