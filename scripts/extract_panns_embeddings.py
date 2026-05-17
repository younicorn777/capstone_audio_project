import os
import numpy as np
import pandas as pd
import librosa

from panns_inference import AudioTagging


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

DATASET_DIR = os.path.join(BASE_DIR, "external_noise_dataset_verified")

RESULT_DIR = os.path.join(BASE_DIR, "results", "ml")
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "panns_embeddings.csv")

SR = 32000

LABELS = ["air", "engine", "dog"]

CHECKPOINT_PATH = os.path.join(
    os.path.expanduser("~"),
    "panns_data",
    "Cnn14_mAP=0.431.pth"
)


# =========================
# 2. 오디오 로드
# =========================

def load_audio_for_panns(path, sr=SR):
    """
    PANNs 입력용 오디오 로드
    - mono
    - 32 kHz
    - float32
    - batch 형태 [1, time]
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)

    # 너무 짧은 파일은 1초까지 padding
    min_len = sr
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    return audio[None, :]


# =========================
# 3. PANNs embedding 추출
# =========================

def extract_embedding(model, audio):
    """
    PANNs embedding 추출

    PANNs output:
    - clipwise_output: [batch, 527]
    - embedding: [batch, 2048]

    파일 하나당 2048차원 embedding vector 사용
    """
    clipwise_output, embedding = model.inference(audio)

    # batch dimension 제거
    embedding_vector = embedding[0]

    return embedding_vector


# =========================
# 4. main
# =========================

def main():
    print("PANNs embedding 추출 시작")
    print("-" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[오류] 데이터셋 폴더가 없습니다: {DATASET_DIR}")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[오류] PANNs checkpoint가 없습니다: {CHECKPOINT_PATH}")
        print("먼저 scripts/setup_panns_files.py를 실행하세요.")
        return

    print("PANNs AudioTagging 모델 로딩 중...")
    model = AudioTagging(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu"
    )

    rows = []

    for label in LABELS:
        label_dir = os.path.join(DATASET_DIR, label)

        if not os.path.exists(label_dir):
            print(f"[경고] label 폴더 없음: {label_dir}")
            continue

        wav_files = sorted([
            f for f in os.listdir(label_dir)
            if f.lower().endswith(".wav")
        ])

        print(f"\n[{label}] 파일 수: {len(wav_files)}")

        for idx, wav_file in enumerate(wav_files, start=1):
            wav_path = os.path.join(label_dir, wav_file)

            try:
                audio = load_audio_for_panns(wav_path)
                embedding = extract_embedding(model, audio)

                row = {
                    "file": wav_file,
                    "label": label,
                }

                for i, value in enumerate(embedding):
                    row[f"emb_{i}"] = float(value)

                rows.append(row)

            except Exception as e:
                print(f"[오류] {wav_file} 처리 실패: {e}")

            if idx % 10 == 0:
                print(f"[{label}] {idx}/{len(wav_files)} 처리 완료")

    if len(rows) == 0:
        print("[오류] 추출된 embedding이 없습니다.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    embedding_cols = [c for c in df.columns if c.startswith("emb_")]

    print("\n" + "-" * 70)
    print("PANNs embedding 추출 완료")
    print(f"저장 위치: {OUTPUT_CSV}")
    print(f"총 샘플 수: {len(df)}")
    print(f"embedding 차원: {len(embedding_cols)}")


if __name__ == "__main__":
    main()