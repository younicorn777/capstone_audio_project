import os
import shutil
import pandas as pd


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

URBANSOUND_DIR = os.path.join(BASE_DIR, "UrbanSound8K")
AUDIO_DIR = os.path.join(URBANSOUND_DIR, "audio")
METADATA_PATH = os.path.join(URBANSOUND_DIR, "metadata", "UrbanSound8K.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "external_noise_dataset_raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. 클래스 매핑
# =========================

# UrbanSound8K class name -> 우리 프로젝트 label
CLASS_MAPPING = {
    "air_conditioner": "air",
    "engine_idling": "engine",
    "dog_bark": "dog",
}


# 클래스별 최대 파일 수
# 너무 많으면 처리 시간이 길어질 수 있으므로 우선 300개씩 사용
MAX_FILES_PER_CLASS = 300

# =========================
# 3. 메인 함수
# =========================

def main():
    print("UrbanSound8K subset 추출 시작")
    print("-" * 60)

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

    total_copied = 0

    for original_class, target_label in CLASS_MAPPING.items():
        print(f"\n[{original_class} → {target_label}] 처리 중")

        target_dir = os.path.join(OUTPUT_DIR, target_label)
        os.makedirs(target_dir, exist_ok=True)

        class_df = metadata[metadata["class"] == original_class].copy()

        if len(class_df) == 0:
            print(f"[경고] 클래스 데이터 없음: {original_class}")
            continue

        # fold 순서와 파일명 기준으로 정렬
        class_df = class_df.sort_values(by=["fold", "slice_file_name"])

        # 최대 파일 수 제한
        class_df = class_df.head(MAX_FILES_PER_CLASS)

        copied_count = 0

        for idx, row in enumerate(class_df.itertuples(index=False), start=1):
            file_name = row.slice_file_name
            fold = row.fold

            source_path = os.path.join(
                AUDIO_DIR,
                f"fold{fold}",
                file_name
            )

            if not os.path.exists(source_path):
                print(f"[경고] 파일 없음: {source_path}")
                continue

            output_name = f"{target_label}_{idx:03d}.wav"
            output_path = os.path.join(target_dir, output_name)

            shutil.copy2(source_path, output_path)

            copied_count += 1
            total_copied += 1

        print(f"[완료] {target_label}: {copied_count}개 복사")

    print("\n" + "-" * 60)
    print("UrbanSound8K subset 추출 완료")
    print(f"총 복사 파일 수: {total_copied}")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()