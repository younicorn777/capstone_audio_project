import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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
    "embedding_separation_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = ["air", "dog", "engine"]


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

    return df, X, y, embedding_cols


# =========================
# 3. 클래스 중심 계산
# =========================

def compute_class_centroids(X, y):
    centroids = {}

    for label in sorted(set(y)):
        X_label = X[y == label]
        centroids[label] = np.mean(X_label, axis=0)

    return centroids


# =========================
# 4. 클래스 내부 거리 계산
# =========================

def compute_intra_class_distance(X, y, centroids):
    """
    각 sample이 자기 class centroid에서 얼마나 떨어져 있는지 평균 거리 계산
    """
    rows = []

    for label in sorted(set(y)):
        X_label = X[y == label]
        centroid = centroids[label]

        distances = np.linalg.norm(X_label - centroid, axis=1)

        rows.append({
            "label": label,
            "intra_distance_mean": float(np.mean(distances)),
            "intra_distance_std": float(np.std(distances)),
            "sample_count": len(X_label)
        })

    return pd.DataFrame(rows)


# =========================
# 5. 클래스 간 중심 거리 계산
# =========================

def compute_inter_class_distance(centroids):
    labels = sorted(centroids.keys())

    rows = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_a = labels[i]
            label_b = labels[j]

            distance = np.linalg.norm(
                centroids[label_a] - centroids[label_b]
            )

            rows.append({
                "class_pair": f"{label_a} - {label_b}",
                "class_a": label_a,
                "class_b": label_b,
                "centroid_distance": float(distance)
            })

    return pd.DataFrame(rows)


# =========================
# 6. 전체 분리도 계산
# =========================

def compute_separation_summary(intra_df, inter_df):
    mean_intra = intra_df["intra_distance_mean"].mean()
    mean_inter = inter_df["centroid_distance"].mean()

    separation_ratio = mean_inter / mean_intra if mean_intra > 0 else np.nan

    return {
        "mean_intra_class_distance": float(mean_intra),
        "mean_inter_class_distance": float(mean_inter),
        "separation_ratio": float(separation_ratio)
    }


# =========================
# 7. 모델별 분석
# =========================

def analyze_embedding(model_name, csv_path):
    print("\n" + "=" * 80)
    print(f"{model_name} embedding 분리도 분석")
    print("=" * 80)

    df, X, y, embedding_cols = load_embedding_data(csv_path)

    print(f"샘플 수: {len(df)}")
    print(f"Embedding 차원: {len(embedding_cols)}")
    print("클래스별 샘플 수:")
    print(df["label"].value_counts())

    # 공정 비교를 위해 scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    centroids = compute_class_centroids(X_scaled, y)

    intra_df = compute_intra_class_distance(
        X_scaled,
        y,
        centroids
    )

    inter_df = compute_inter_class_distance(
        centroids
    )

    summary = compute_separation_summary(
        intra_df,
        inter_df
    )

    print("\n[클래스 내부 거리]")
    print(intra_df)

    print("\n[클래스 간 중심 거리]")
    print(inter_df)

    print("\n[전체 분리도 요약]")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    # 결과 저장
    prefix = model_name.lower()

    intra_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_intra_class_distance.csv"
    )

    inter_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_inter_class_distance.csv"
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"{prefix}_separation_summary.csv"
    )

    intra_df.to_csv(intra_path, index=False, encoding="utf-8-sig")
    inter_df.to_csv(inter_path, index=False, encoding="utf-8-sig")

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "model": model_name,
        **summary
    }


# =========================
# 8. main
# =========================

def main():
    print("Embedding 클래스 분리도 분석 시작")

    results = []

    yamnet_summary = analyze_embedding(
        model_name="YAMNet",
        csv_path=YAMNET_EMBEDDING_CSV
    )

    panns_summary = analyze_embedding(
        model_name="PANNs",
        csv_path=PANNS_EMBEDDING_CSV
    )

    results.append(yamnet_summary)
    results.append(panns_summary)

    summary_df = pd.DataFrame(results)

    summary_csv = os.path.join(
        OUTPUT_DIR,
        "embedding_separation_summary_all.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("YAMNet vs PANNs embedding 분리도 비교")
    print("=" * 80)
    print(summary_df)

    print(f"\n전체 요약 저장: {summary_csv}")


if __name__ == "__main__":
    main()