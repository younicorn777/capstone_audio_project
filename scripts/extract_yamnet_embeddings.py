import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

DATASET_DIR = os.path.join(BASE_DIR, "external_noise_dataset_verified")

RESULT_DIR = os.path.join(BASE_DIR, "results", "ml")
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "yamnet_embeddings.csv")

SR = 16000

LABELS = ["air", "engine", "dog"]


# =========================
# 2. 오디오 로드 함수
# =========================

def load_audio_for_yamnet(path, sr=SR):
    """
    YAMNet 입력용 오디오 로드
    - mono
    - 16 kHz
    - float32
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)

    # 너무 짧은 오디오는 1초까지 padding
    min_len = sr
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    return audio


# =========================
# 3. YAMNet embedding 추출
# =========================

def extract_embedding(model, audio):
    """
    YAMNet embedding 추출

    YAMNet output:
    - scores: [frames, 521]
    - embeddings: [frames, 1024]
    - spectrogram

    파일 하나에 대해 여러 frame embedding이 나오므로,
    평균을 내서 파일 하나당 1024차원 vector로 변환
    """
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

    scores, embeddings, spectrogram = model(waveform)

    embeddings_np = embeddings.numpy()

    # frame 평균 embedding
    mean_embedding = np.mean(embeddings_np, axis=0)

    return mean_embedding


# =========================
# 4. 전체 데이터셋 처리
# =========================

def main():
    print("YAMNet embedding 추출 시작")
    print("-" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[오류] 데이터셋 폴더가 없습니다: {DATASET_DIR}")
        return

    print("YAMNet 모델 로딩 중...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

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
                audio = load_audio_for_yamnet(wav_path)
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

    print("\n" + "-" * 70)
    print("YAMNet embedding 추출 완료")
    print(f"저장 위치: {OUTPUT_CSV}")
    print(f"총 샘플 수: {len(df)}")
    print(f"embedding 차원: {len([c for c in df.columns if c.startswith('emb_')])}")


if __name__ == "__main__":
    main()