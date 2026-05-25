import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.svm import SVC
from xgboost import XGBClassifier


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

YAMNET_EMBEDDING_CSV = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "yamnet_embeddings.csv"
)

PANNS_EMBEDDING_CSV = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "panns_embeddings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "cross_validation_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# =========================
# 2. 데이터 로드
# =========================

def load_embedding_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embedding CSV 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    embedding_cols = [c for c in df.columns if c.startswith("emb_")]

    if len(embedding_cols) == 0:
        raise ValueError("embedding column이 없습니다.")

    X = df[embedding_cols].values
    y = df["label"].values

    return X, y


# =========================
# 3. Cross Validation
# =========================

def evaluate_cv(model_name, model, X, y):
    print("\n" + "=" * 80)
    print(f"{model_name} 5-Fold Cross Validation")
    print("=" * 80)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro"
    }

    scores = cross_validate(
        model,
        X,
        y_encoded,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    result = {
        "Model": model_name,

        "Accuracy Mean": np.mean(scores["test_accuracy"]),
        "Accuracy Std": np.std(scores["test_accuracy"]),

        "Macro F1 Mean": np.mean(scores["test_f1_macro"]),
        "Macro F1 Std": np.std(scores["test_f1_macro"]),

        "Macro Precision Mean": np.mean(scores["test_precision_macro"]),
        "Macro Precision Std": np.std(scores["test_precision_macro"]),

        "Macro Recall Mean": np.mean(scores["test_recall_macro"]),
        "Macro Recall Std": np.std(scores["test_recall_macro"]),
    }

    print("\n[Fold별 Accuracy]")
    print(scores["test_accuracy"])

    print("\n[Fold별 Macro F1]")
    print(scores["test_f1_macro"])

    print("\n[요약]")
    for k, v in result.items():
        if k == "Model":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")

    return result


# =========================
# 4. main
# =========================

def main():
    print("Noise classifier 5-Fold Cross Validation 시작")

    results = []

    # -------------------------
    # YAMNet
    # -------------------------

    X_yamnet, y_yamnet = load_embedding_data(YAMNET_EMBEDDING_CSV)

    yamnet_models = {
        "YAMNet + SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

        "YAMNet + XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )
    }

    for model_name, model in yamnet_models.items():
        result = evaluate_cv(
            model_name=model_name,
            model=model,
            X=X_yamnet,
            y=y_yamnet
        )

        results.append(result)

    # -------------------------
    # PANNs
    # -------------------------

    X_panns, y_panns = load_embedding_data(PANNS_EMBEDDING_CSV)

    panns_models = {
        "PANNs + SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

        "PANNs + XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )
    }

    for model_name, model in panns_models.items():
        result = evaluate_cv(
            model_name=model_name,
            model=model,
            X=X_panns,
            y=y_panns
        )

        results.append(result)

    # -------------------------
    # 결과 저장
    # -------------------------

    results_df = pd.DataFrame(results)

    output_csv = os.path.join(
        OUTPUT_DIR,
        "5fold_cross_validation_results.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("전체 Cross Validation 결과")
    print("=" * 80)

    print(results_df)

    print(f"\n결과 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()