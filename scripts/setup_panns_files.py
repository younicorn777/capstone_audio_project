import os
import urllib.request


PANNs_DIR = os.path.join(os.path.expanduser("~"), "panns_data")
os.makedirs(PANNs_DIR, exist_ok=True)

LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

LABELS_PATH = os.path.join(PANNs_DIR, "class_labels_indices.csv")
CHECKPOINT_PATH = os.path.join(PANNs_DIR, "Cnn14_mAP=0.431.pth")


def download_if_missing(url, path):
    if os.path.exists(path):
        print(f"[이미 존재] {path}")
        return

    print(f"[다운로드 시작] {url}")
    print(f"[저장 위치] {path}")

    urllib.request.urlretrieve(url, path)

    print(f"[다운로드 완료] {path}")


def main():
    print("PANNs 필요 파일 다운로드 시작")
    print(f"PANNs data folder: {PANNs_DIR}")
    print("-" * 70)

    download_if_missing(LABELS_URL, LABELS_PATH)
    download_if_missing(CHECKPOINT_URL, CHECKPOINT_PATH)

    print("-" * 70)
    print("PANNs 필요 파일 준비 완료")
    print(f"Label CSV: {LABELS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()