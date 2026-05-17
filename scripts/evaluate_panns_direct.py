import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from panns_inference import AudioTagging, labels

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

DATASET_DIR = os.path.join(BASE_DIR, "external_noise_dataset_verified")

RESULT_DIR = os.path.join(BASE_DIR, "results", "ml", "panns_direct")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SUMMARY_JSON = os.path.join(RESULT_DIR, "panns_direct_summary.json")
PREDICTION_CSV = os.path.join(RESULT_DIR, "panns_direct_predictions.csv")

SR = 32000

LABELS = ["air", "dog", "engine"]

CHECKPOINT_PATH = os.path.join(
    os.path.expanduser("~"),
    "panns_data",
    "Cnn14_mAP=0.431.pth"
)


# =========================
# 2. PANNs target keyword
# =========================

TARGET_KEYWORDS = {
    "air": [
        "air conditioning",
        "air conditioner",
        "air",
        "fan",
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
# 3. 오디오 로드
# =========================

def load_audio_for_panns(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)

    # 너무 짧은 파일은 1초까지 padding
    min_len = sr
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    # PANNs 입력 형태: [batch, time]
    audio = audio[None, :]

    return audio


# =========================
# 4. target class index 찾기
# =========================

def get_target_indices(panns_labels, keywords):
    indices = []

    for i, name in enumerate(panns_labels):
        name_lower = str(name).lower()

        for keyword in keywords:
            if keyword.lower() in name_lower:
                indices.append(i)
                break

    return sorted(set(indices))


def print_target_classes(label, target_indices):
    print(f"\n[{label}] PANNs target class index 수: {len(target_indices)}")

    if len(target_indices) == 0:
        print(" - 없음")
        return

    for idx in target_indices:
        print(f" - {idx}: {labels[idx]}")


# =========================
# 5. class score 계산
# =========================

def compute_class_score(scores, target_indices):
    """
    scores:
        PANNs clipwise_output[0]
    target_indices:
        air / dog / engine 관련 PANNs class index
    """
    if len(target_indices) == 0:
        return 0.0

    return float(np.max(scores[target_indices]))


# =========================
# 6. Confusion Matrix Plot
# =========================

def plot_confusion_matrix(cm, labels_list, title, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels_list))
    plt.xticks(tick_marks, labels_list, rotation=45, ha="right")
    plt.yticks(tick_marks, labels_list)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=14,          # 글씨 크기 키우기
                fontweight="bold"     # 글씨 두껍게
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# 7. main
# =========================

def main():
    print("PANNs Direct 예측 평가 시작")
    print("-" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[오류] 데이터셋 폴더가 없습니다: {DATASET_DIR}")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[오류] PANNs checkpoint가 없습니다: {CHECKPOINT_PATH}")
        print("먼저 scripts/setup_panns_files.py를 실행하세요.")
        return

    print("PANNs AudioTagging 모델 로딩 중...")
    at = AudioTagging(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu"
    )

    # label별 PANNs class index 계산
    label_to_indices = {}

    print("\n사용되는 PANNs class")

    for label in LABELS:
        indices = get_target_indices(
            panns_labels=labels,
            keywords=TARGET_KEYWORDS[label]
        )

        label_to_indices[label] = indices
        print_target_classes(label, indices)

    y_true = []
    y_pred = []
    rows = []

    # =========================
    # 데이터셋 평가
    # =========================

    for true_label in LABELS:
        label_dir = os.path.join(DATASET_DIR, true_label)

        if not os.path.exists(label_dir):
            print(f"[경고] label 폴더 없음: {label_dir}")
            continue

        wav_files = sorted([
            f for f in os.listdir(label_dir)
            if f.lower().endswith(".wav")
        ])

        print(f"\n[{true_label}] 파일 수: {len(wav_files)}")

        for idx, wav_file in enumerate(wav_files, start=1):
            wav_path = os.path.join(label_dir, wav_file)

            try:
                audio = load_audio_for_panns(wav_path)

                clipwise_output, embedding = at.inference(audio)

                scores = clipwise_output[0]

                label_scores = {}

                for candidate_label in LABELS:
                    score = compute_class_score(
                        scores=scores,
                        target_indices=label_to_indices[candidate_label]
                    )

                    label_scores[candidate_label] = score

                predicted_label = max(
                    label_scores,
                    key=label_scores.get
                )

                y_true.append(true_label)
                y_pred.append(predicted_label)

                row = {
                    "file": wav_file,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }

                for key, value in label_scores.items():
                    row[f"{key}_score"] = value

                rows.append(row)

            except Exception as e:
                print(f"[오류] {wav_file} 처리 실패: {e}")

            if idx % 10 == 0:
                print(f"[{true_label}] {idx}/{len(wav_files)} 처리 완료")

    # =========================
    # 평가
    # =========================

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    cm_path = os.path.join(
        PLOT_DIR,
        "panns_direct_confusion_matrix.png"
    )

    plot_confusion_matrix(
        cm=cm,
        labels_list=LABELS,
        title="PANNs Direct Confusion Matrix",
        output_path=cm_path,
    )

    # =========================
    # 결과 저장
    # =========================

    pd.DataFrame(rows).to_csv(
        PREDICTION_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "model": "PANNs Direct",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
        "classification_report": report_dict,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # =========================
    # 출력
    # =========================

    print("\n" + "-" * 70)
    print("PANNs Direct 예측 평가 완료")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    print("\nClassification Report")
    print(report_text)

    print(f"\nConfusion Matrix 저장: {cm_path}")
    print(f"Prediction CSV 저장: {PREDICTION_CSV}")
    print(f"Summary JSON 저장: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()