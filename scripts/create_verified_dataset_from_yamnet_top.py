import os
import shutil


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

# YAMNet이 상위 후보를 복사해둔 폴더
FILTERED_DIR = os.path.join(BASE_DIR, "yamnet_filtered_candidates")

# 최종 학습에 사용할 verified dataset 폴더
VERIFIED_DIR = os.path.join(BASE_DIR, "external_noise_dataset_verified")

LABELS = ["air", "engine", "dog"]

# 클래스별 사용할 파일 수
TOP_N_PER_CLASS = 50


# =========================
# 2. 폴더 초기화 함수
# =========================

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_label_folder(label_dir):
    """
    기존 verified 폴더가 있을 경우,
    이전 파일과 섞이지 않도록 해당 label 폴더 안의 wav 파일만 삭제
    """
    if not os.path.exists(label_dir):
        os.makedirs(label_dir, exist_ok=True)
        return

    for file_name in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file_name)

        if os.path.isfile(file_path) and file_name.lower().endswith(".wav"):
            os.remove(file_path)


# =========================
# 3. 파일 정렬 함수
# =========================

def get_sorted_wav_files(label_dir):
    """
    yamnet_filtered_candidates 안의 파일은 보통 아래 형식임:
    air_rank001_score0.8123_원본파일명.wav

    따라서 파일명 정렬을 하면 rank001, rank002 순서로 정렬됨.
    """
    wav_files = [
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".wav")
    ]

    wav_files = sorted(wav_files)

    return wav_files


# =========================
# 4. verified dataset 생성
# =========================

def main():
    print("YAMNet 상위 후보 기반 verified dataset 생성 시작")
    print(f"클래스별 선택 파일 수: {TOP_N_PER_CLASS}")
    print("-" * 70)

    if not os.path.exists(FILTERED_DIR):
        print(f"[오류] YAMNet 후보 폴더가 없습니다: {FILTERED_DIR}")
        return

    make_dir(VERIFIED_DIR)

    total_copied = 0
    copied_summary = {}

    for label in LABELS:
        source_label_dir = os.path.join(FILTERED_DIR, label)
        target_label_dir = os.path.join(VERIFIED_DIR, label)

        if not os.path.exists(source_label_dir):
            print(f"[경고] 후보 label 폴더가 없습니다: {source_label_dir}")
            copied_summary[label] = 0
            continue

        make_dir(target_label_dir)

        # 기존 파일과 섞이지 않도록 초기화
        clear_label_folder(target_label_dir)

        wav_files = get_sorted_wav_files(source_label_dir)

        if len(wav_files) == 0:
            print(f"[경고] wav 파일이 없습니다: {source_label_dir}")
            copied_summary[label] = 0
            continue

        selected_files = wav_files[:TOP_N_PER_CLASS]

        copied_count = 0

        for idx, file_name in enumerate(selected_files, start=1):
            src = os.path.join(source_label_dir, file_name)

            # 학습용 폴더에서는 파일명을 단순화
            # 예: air_001.wav, engine_001.wav, dog_001.wav
            dst_name = f"{label}_{idx:03d}.wav"
            dst = os.path.join(target_label_dir, dst_name)

            shutil.copy2(src, dst)
            copied_count += 1

        copied_summary[label] = copied_count
        total_copied += copied_count

        print(f"[완료] {label}: {copied_count}개 복사")

    print("-" * 70)
    print("verified dataset 생성 완료")
    print(f"저장 위치: {VERIFIED_DIR}")
    print(f"총 복사 파일 수: {total_copied}")

    print("\n클래스별 파일 수")
    for label, count in copied_summary.items():
        print(f"- {label}: {count}개")

    if total_copied == TOP_N_PER_CLASS * len(LABELS):
        print("\n정상 완료: 클래스별 50개씩 총 150개 생성됨")
    else:
        print("\n주의: 일부 클래스의 파일 수가 50개보다 적을 수 있음")


if __name__ == "__main__":
    main()