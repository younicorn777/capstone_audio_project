import os
import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

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

RESULT_DIR = os.path.join(BASE_DIR, "results", "ml", "yamnet_direct")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SUMMARY_JSON = os.path.join(RESULT_DIR, "yamnet_direct_summary.json")

SR = 16000

LABELS = ["air", "dog", "engine"]


# =========================
# 2. YAMNet target keyword
# =========================

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
# 3. 오디오 로드
# =========================

def load_audio_for_yamnet(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)

    min_len = sr
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    return audio


# =========================
# 4. target class index 찾기
# =========================

def get_target_indices(class_names, keywords):
    indices = []

    for i, name in enumerate(class_names):
        name_lower = str(name).lower()

        for keyword in keywords:
            if keyword.lower() in name_lower:
                indices.append(i)
                break

    return sorted(set(indices))


# =========================
# 5. class score 계산
# =========================

def compute_class_score(mean_scores, target_indices):
    if len(target_indices) == 0:
        return 0.0

    return float(np.max(mean_scores[target_indices]))


# =========================
# 6. Confusion Matrix Plot
# =========================

def plot_confusion_matrix(cm, labels, title, output_path):
    plt.figure(figsize=(7, 6))

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

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
    print("YAMNet 단독 예측 평가 시작")
    print("-" * 70)

    if not os.path.exists(DATASET_DIR):
        print(f"[오류] 데이터셋 폴더가 없습니다: {DATASET_DIR}")
        return

    print("YAMNet 모델 로딩 중...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_map = pd.read_csv(class_map_path)

    class_names = class_map["display_name"].tolist()

    # label별 YAMNet class index 계산
    label_to_indices = {}

    print("\n사용되는 YAMNet class")

    for label in LABELS:
        indices = get_target_indices(
            class_names=class_names,
            keywords=TARGET_KEYWORDS[label]
        )

        label_to_indices[label] = indices

        print(f"\n[{label}]")

        for idx in indices:
            print(f"- {idx}: {class_names[idx]}")

    y_true = []
    y_pred = []

    rows = []

    # =========================
    # 데이터셋 평가
    # =========================

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

                waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

                scores, embeddings, spectrogram = model(waveform)

                scores_np = scores.numpy()

                # frame 평균 score
                mean_scores = np.mean(scores_np, axis=0)

                # air / dog / engine score 계산
                label_scores = {}

                for candidate_label in LABELS:
                    score = compute_class_score(
                        mean_scores,
                        label_to_indices[candidate_label]
                    )

                    label_scores[candidate_label] = score

                # 가장 score 높은 label 선택
                predicted_label = max(
                    label_scores,
                    key=label_scores.get
                )

                y_true.append(label)
                y_pred.append(predicted_label)

                row = {
                    "file": wav_file,
                    "true_label": label,
                    "predicted_label": predicted_label,
                }

                # 각 label score 저장
                for key, value in label_scores.items():
                    row[f"{key}_score"] = value

                rows.append(row)

            except Exception as e:
                print(f"[오류] {wav_file} 처리 실패: {e}")

            if idx % 10 == 0:
                print(f"[{label}] {idx}/{len(wav_files)} 처리 완료")

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
        "yamnet_direct_confusion_matrix.png"
    )

    plot_confusion_matrix(
        cm=cm,
        labels=LABELS,
        title="YAMNet Direct Confusion Matrix",
        output_path=cm_path,
    )

    # =========================
    # 결과 저장
    # =========================

    detail_csv = os.path.join(
        RESULT_DIR,
        "yamnet_direct_predictions.csv"
    )

    pd.DataFrame(rows).to_csv(
        detail_csv,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "model": "YAMNet Direct",
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
    print("YAMNet 단독 예측 평가 완료")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    print("\nClassification Report")
    print(report_text)

    print(f"\nConfusion Matrix 저장: {cm_path}")
    print(f"Prediction CSV 저장: {detail_csv}")
    print(f"Summary JSON 저장: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
