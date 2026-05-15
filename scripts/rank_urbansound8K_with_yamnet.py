import os
import shutil
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

URBANSOUND_DIR = os.path.join(BASE_DIR, "UrbanSound8K")
AUDIO_DIR = os.path.join(URBANSOUND_DIR, "audio")
METADATA_PATH = os.path.join(URBANSOUND_DIR, "metadata", "UrbanSound8K.csv")

RESULT_DIR = os.path.join(BASE_DIR, "yamnet_candidate_ranking")
FILTERED_DIR = os.path.join(BASE_DIR, "yamnet_filtered_candidates")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

SR = 16000

# 클래스별 상위 몇 개를 팀원 검수 후보로 복사할지
TOP_N_PER_CLASS = 100


# =========================
# 2. UrbanSound8K class → 프로젝트 label 매핑
# =========================

CLASS_MAPPING = {
    "air_conditioner": "air",
    "engine_idling": "engine",
    "dog_bark": "dog",
}


# =========================
# 3. YAMNet target keyword 설정
# =========================
# YAMNet class 이름이 UrbanSound8K class와 완전히 같지 않을 수 있으므로
# 관련 keyword가 포함된 YAMNet class score를 사용함.

TARGET_KEYWORDS = {
    "air": [
        "air conditioning",
        "air conditioner",
        "fan",
        "air",
    ],
    "engine": [
        "engine",
        "vehicle",
        "car",
        "motor",
        "truck",
    ],
    "dog": [
        "dog",
        "bark",
    ],
}


# =========================
# 4. 오디오 로드
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

    # YAMNet 입력이 너무 짧으면 안정성이 떨어질 수 있어 최소 1초 padding
    min_len = sr
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    return audio


# =========================
# 5. YAMNet class index 찾기
# =========================

def get_target_indices(class_names, keywords):
    """
    YAMNet class display_name 중 keyword를 포함하는 class index 찾기
    """
    indices = []

    for i, name in enumerate(class_names):
        name_lower = str(name).lower()

        for keyword in keywords:
            if keyword.lower() in name_lower:
                indices.append(i)
                break

    return sorted(set(indices))


def print_target_classes(label, class_names, target_indices):
    print(f"\n[{label}] YAMNet target class index 수: {len(target_indices)}")
    print("[사용된 YAMNet class name]")

    if len(target_indices) == 0:
        print(" - 없음")
        return

    for idx in target_indices:
        print(f" - {idx}: {class_names[idx]}")


# =========================
# 6. 파일별 YAMNet score 계산
# =========================

def compute_yamnet_score(model, audio, target_indices):
    """
    target class index들에 대한 score 계산

    YAMNet output:
    - scores: [frames, 521]
    - embeddings: [frames, 1024]
    - spectrogram

    여기서는 파일 전체 frame 평균 score를 구한 뒤,
    target 관련 class 중 가장 높은 score를 사용함.
    """
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

    scores, embeddings, spectrogram = model(waveform)
    scores_np = scores.numpy()

    mean_scores = np.mean(scores_np, axis=0)

    if len(target_indices) == 0:
        return 0.0

    target_score = np.max(mean_scores[target_indices])
    return float(target_score)


# =========================
# 7. UrbanSound8K 후보 ranking
# =========================

def get_source_path(row):
    """
    UrbanSound8K metadata row에서 실제 wav path 생성
    """
    file_name = row["slice_file_name"]
    fold = row["fold"]

    return os.path.join(
        AUDIO_DIR,
        f"fold{fold}",
        file_name
    )


def rank_candidates_for_class(model, class_names, metadata, urban_class, target_label):
    """
    UrbanSound8K metadata에서 특정 class만 선택 후
    YAMNet score로 ranking
    """
    class_df = metadata[metadata["class"] == urban_class].copy()

    if len(class_df) == 0:
        print(f"[경고] metadata에서 class를 찾지 못함: {urban_class}")
        return []

    target_indices = get_target_indices(
        class_names=class_names,
        keywords=TARGET_KEYWORDS[target_label]
    )

    print_target_classes(target_label, class_names, target_indices)

    rows = []

    print(f"\n[{urban_class} → {target_label}] 후보 파일 수: {len(class_df)}")

    for idx, (_, row) in enumerate(class_df.iterrows(), start=1):
        source_path = get_source_path(row)

        if not os.path.exists(source_path):
            print(f"[경고] 파일 없음: {source_path}")
            continue

        try:
            audio = load_audio_for_yamnet(source_path)
            score = compute_yamnet_score(model, audio, target_indices)

            rows.append({
                "project_label": target_label,
                "urbansound_class": urban_class,
                "original_file": row["slice_file_name"],
                "fold": int(row["fold"]),
                "yamnet_target_score": score,
                "source_path": source_path,
            })

        except Exception as e:
            print(f"[오류] 처리 실패: {source_path} / {e}")

        if idx % 50 == 0:
            print(f"[{target_label}] {idx}/{len(class_df)} 처리 완료")

    rows = sorted(
        rows,
        key=lambda x: x["yamnet_target_score"],
        reverse=True
    )

    return rows


# =========================
# 8. 상위 후보 복사
# =========================

def copy_top_candidates(rows, target_label, top_n=TOP_N_PER_CLASS):
    """
    ranking 상위 파일을 yamnet_filtered_candidates/{label}/로 복사
    """
    output_label_dir = os.path.join(FILTERED_DIR, target_label)
    os.makedirs(output_label_dir, exist_ok=True)

    top_rows = rows[:top_n]

    copied = 0

    for rank, row in enumerate(top_rows, start=1):
        src = row["source_path"]
        original_name = row["original_file"]

        # 검수자가 듣기 편하도록 score와 rank를 파일명에 표시
        # 예: air_rank001_score0.8123_100852-0-0-0.wav
        dst_name = (
            f"{target_label}_rank{rank:03d}_"
            f"score{row['yamnet_target_score']:.4f}_"
            f"{original_name}"
        )

        dst = os.path.join(output_label_dir, dst_name)

        shutil.copy2(src, dst)
        copied += 1

    print(f"[복사 완료] {target_label}: 상위 {copied}개 → {output_label_dir}")


# =========================
# 9. main
# =========================

def main():
    print("UrbanSound8K 원본 기반 YAMNet 후보 ranking 시작")
    print("-" * 70)

    if not os.path.exists(METADATA_PATH):
        print(f"[오류] metadata 파일이 없습니다: {METADATA_PATH}")
        return

    if not os.path.exists(AUDIO_DIR):
        print(f"[오류] audio 폴더가 없습니다: {AUDIO_DIR}")
        return

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = {"slice_file_name", "fold", "class"}
    if not required_columns.issubset(metadata.columns):
        print("[오류] UrbanSound8K.csv에 필요한 컬럼이 없습니다.")
        print(f"필요 컬럼: {required_columns}")
        print(f"현재 컬럼: {metadata.columns.tolist()}")
        return

    print("YAMNet 모델 로딩 중...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_map = pd.read_csv(class_map_path)
    class_names = class_map["display_name"].tolist()

    all_rows = []

    for urban_class, target_label in CLASS_MAPPING.items():
        rows = rank_candidates_for_class(
            model=model,
            class_names=class_names,
            metadata=metadata,
            urban_class=urban_class,
            target_label=target_label
        )

        if len(rows) == 0:
            continue

        all_rows.extend(rows)

        # 클래스별 ranking CSV 저장
        ranking_csv = os.path.join(
            RESULT_DIR,
            f"{target_label}_yamnet_ranking.csv"
        )
        pd.DataFrame(rows).to_csv(
            ranking_csv,
            index=False,
            encoding="utf-8-sig"
        )

        # 상위 후보 복사
        copy_top_candidates(
            rows=rows,
            target_label=target_label,
            top_n=TOP_N_PER_CLASS
        )

    # 전체 ranking CSV 저장
    all_csv = os.path.join(RESULT_DIR, "all_yamnet_ranking.csv")
    pd.DataFrame(all_rows).to_csv(
        all_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("-" * 70)
    print("YAMNet 후보 ranking 완료")
    print(f"Ranking CSV 저장 위치: {RESULT_DIR}")
    print(f"상위 후보 파일 저장 위치: {FILTERED_DIR}")


if __name__ == "__main__":
    main()