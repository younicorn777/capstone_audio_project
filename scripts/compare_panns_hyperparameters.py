import os
import itertools
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

INPUT_CSV = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "panns_embeddings.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "panns_hyperparameter_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# =========================
# 2. 데이터 로드
# =========================

def load_embedding_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embedding CSV 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    embedding_cols = [
        c for c in df.columns
        if c.startswith("emb_")
    ]

    if len(embedding_cols) == 0:
        raise ValueError("embedding column이 없습니다.")

    X = df[embedding_cols].values
    y_text = df["label"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    print(f"전체 샘플 수: {len(df)}")
    print(f"Embedding 차원: {len(embedding_cols)}")
    print("클래스 매핑:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print("클래스별 샘플 수:")
    print(df["label"].value_counts())

    return X, y


# =========================
# 3. CV 평가 함수
# =========================

def run_cv(model, X, y):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
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
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "macro_f1_mean": float(np.mean(scores["test_f1_macro"])),
        "macro_f1_std": float(np.std(scores["test_f1_macro"])),
        "macro_precision_mean": float(np.mean(scores["test_precision_macro"])),
        "macro_precision_std": float(np.std(scores["test_precision_macro"])),
        "macro_recall_mean": float(np.mean(scores["test_recall_macro"])),
        "macro_recall_std": float(np.std(scores["test_recall_macro"])),
    }


# =========================
# 4. SVM 하이퍼파라미터 실험
# =========================

def experiment_svm(X, y):
    print("\n" + "=" * 80)
    print("PANNs + SVM 하이퍼파라미터 실험")
    print("=" * 80)

    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
    }

    rows = []

    for C, gamma in itertools.product(
        param_grid["C"],
        param_grid["gamma"]
    ):
        print(f"\n[SVM] C={C}, gamma={gamma}")

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=RANDOM_STATE,
            ))
        ])

        result = run_cv(model, X, y)

        row = {
            "model": "PANNs + SVM",
            "C": C,
            "gamma": gamma,
            **result
        }

        rows.append(row)

        print(
            f"Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}, "
            f"Macro F1: {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}"
        )

    df = pd.DataFrame(rows)

    output_csv = os.path.join(
        OUTPUT_DIR,
        "svm_hyperparameter_results.csv"
    )

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    best = df.sort_values(
        by=["macro_f1_mean", "accuracy_mean"],
        ascending=False
    ).iloc[0]

    print("\n[SVM Best]")
    print(best)

    return df


# =========================
# 5. XGBoost 하이퍼파라미터 실험
# =========================

def experiment_xgboost(X, y):
    print("\n" + "=" * 80)
    print("PANNs + XGBoost 하이퍼파라미터 실험")
    print("=" * 80)

    param_grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100],
        "subsample": [0.8],
    }

    rows = []

    for max_depth, learning_rate, n_estimators, subsample in itertools.product(
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["n_estimators"],
        param_grid["subsample"],
    ):
        print(
            f"\n[XGBoost] max_depth={max_depth}, "
            f"learning_rate={learning_rate}, "
            f"n_estimators={n_estimators}, "
            f"subsample={subsample}"
        )

        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )

        result = run_cv(model, X, y)

        row = {
            "model": "PANNs + XGBoost",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            **result
        }

        rows.append(row)

        print(
            f"Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}, "
            f"Macro F1: {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}"
        )

    df = pd.DataFrame(rows)

    output_csv = os.path.join(
        OUTPUT_DIR,
        "xgboost_hyperparameter_results.csv"
    )

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    best = df.sort_values(
        by=["macro_f1_mean", "accuracy_mean"],
        ascending=False
    ).iloc[0]

    print("\n[XGBoost Best]")
    print(best)

    return df


# =========================
# 6. Best 모델 비교
# =========================

def summarize_best_results(svm_df, xgb_df):
    svm_best = svm_df.sort_values(
        by=["macro_f1_mean", "accuracy_mean"],
        ascending=False
    ).iloc[0].to_dict()

    xgb_best = xgb_df.sort_values(
        by=["macro_f1_mean", "accuracy_mean"],
        ascending=False
    ).iloc[0].to_dict()

    summary_df = pd.DataFrame([svm_best, xgb_best])

    output_csv = os.path.join(
        OUTPUT_DIR,
        "best_hyperparameter_comparison.csv"
    )

    summary_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("Best Hyperparameter Comparison")
    print("=" * 80)
    print(summary_df)

    print(f"\n저장 위치: {output_csv}")

    return summary_df


# =========================
# 7. main
# =========================

def main():
    print("PANNs embedding 기반 SVM/XGBoost 하이퍼파라미터 비교 실험 시작")
    print("-" * 80)

    X, y = load_embedding_data(INPUT_CSV)

    svm_df = experiment_svm(X, y)
    xgb_df = experiment_xgboost(X, y)

    summarize_best_results(svm_df, xgb_df)

    print("\n실험 완료")
    print(f"결과 저장 폴더: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()