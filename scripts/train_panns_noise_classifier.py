import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

INPUT_CSV = os.path.join(BASE_DIR, "results", "ml", "panns_embeddings.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results", "ml", "panns_classification")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULT_DIR, "panns_classification_summary.csv")
DETAIL_JSON = os.path.join(RESULT_DIR, "panns_classification_report.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================
# 2. 데이터 로드
# =========================

def load_embedding_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embedding CSV 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    embedding_cols = [col for col in df.columns if col.startswith("emb_")]

    if len(embedding_cols) == 0:
        raise ValueError("embedding column이 없습니다. emb_0, emb_1 ... 형식인지 확인하세요.")

    X = df[embedding_cols].values
    y = df["label"].values
    files = df["file"].values

    return df, X, y, files, embedding_cols


# =========================
# 3. Confusion Matrix Plot
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
# 4. 모델 평가 함수
# =========================

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, label_names):
    print(f"\n[{model_name}] 학습 시작")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    cm_path = os.path.join(
        PLOT_DIR,
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plot_confusion_matrix(
        cm=cm,
        labels=label_names,
        title=f"PANNs + {model_name} Confusion Matrix",
        output_path=cm_path,
    )

    print(f"[{model_name}] 평가 결과")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(report_text)
    print(f"Confusion Matrix 저장: {cm_path}")

    summary_row = {
        "model": f"PANNs + {model_name}",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "confusion_matrix_path": cm_path,
    }

    detail = {
        "model": f"PANNs + {model_name}",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": cm_path,
    }

    return summary_row, detail


# =========================
# 5. main
# =========================

def main():
    print("PANNs embedding 기반 소음 환경 분류 모델 학습 시작")
    print("-" * 70)

    df, X, y_text, files, embedding_cols = load_embedding_data(INPUT_CSV)

    print(f"전체 샘플 수: {len(df)}")
    print(f"Embedding 차원: {len(embedding_cols)}")
    print("클래스별 샘플 수:")
    print(df["label"].value_counts())

    # label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    label_names = label_encoder.classes_.tolist()

    print(f"Label mapping: {dict(zip(label_names, range(len(label_names))))}")

    # train / test split
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X,
        y,
        files,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # =========================
    # 6. 모델 정의
    # =========================

    models = {}

    # SVM: 고차원 embedding이므로 scaling 포함
    models["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    # XGBoost: YAMNet 실험과 동일 조건으로 비교
    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )

    summary_rows = []
    detail_results = {}

    # =========================
    # 7. 학습 및 평가
    # =========================

    for model_name, model in models.items():
        summary_row, detail = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_names=label_names,
        )

        summary_rows.append(summary_row)
        detail_results[f"PANNs + {model_name}"] = detail

    # =========================
    # 8. 결과 저장
    # =========================

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    with open(DETAIL_JSON, "w", encoding="utf-8") as f:
        json.dump(detail_results, f, indent=4, ensure_ascii=False)

    print("\n" + "-" * 70)
    print("PANNs 소음 환경 분류 모델 학습 완료")
    print(f"요약 결과 CSV: {SUMMARY_CSV}")
    print(f"상세 결과 JSON: {DETAIL_JSON}")
    print(f"Confusion Matrix 저장 폴더: {PLOT_DIR}")


if __name__ == "__main__":
    main()