import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

INPUT_CSV = os.path.join(BASE_DIR, "results", "ml", "panns_embeddings.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results", "ml", "panns_visualization")
os.makedirs(RESULT_DIR, exist_ok=True)

TSNE_PLOT_PATH = os.path.join(RESULT_DIR, "panns_embedding_tsne.png")
TSNE_CSV_PATH = os.path.join(RESULT_DIR, "panns_embedding_tsne_coordinates.csv")

RANDOM_STATE = 42


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
    labels = df["label"].values
    files = df["file"].values

    return df, X, labels, files, embedding_cols


# =========================
# 3. 시각화 함수
# =========================

def plot_2d_embedding(coords, labels, title, output_path):
    unique_labels = sorted(set(labels))

    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        idx = labels == label

        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=label,
            alpha=0.8,
            s=45
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# 4. t-SNE 시각화
# =========================

def run_tsne(X_scaled, labels, files):
    print("PANNs t-SNE 시각화 생성 중...")

    n_samples = X_scaled.shape[0]

    # 데이터 수에 맞춰 perplexity 자동 조정
    perplexity = min(30, max(5, (n_samples - 1) // 3))

    print(f"t-SNE perplexity: {perplexity}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE
    )

    coords = tsne.fit_transform(X_scaled)

    plot_2d_embedding(
        coords=coords,
        labels=labels,
        title="PANNs Embedding t-SNE",
        output_path=TSNE_PLOT_PATH
    )

    tsne_df = pd.DataFrame({
        "file": files,
        "label": labels,
        "TSNE1": coords[:, 0],
        "TSNE2": coords[:, 1],
    })

    tsne_df.to_csv(TSNE_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"t-SNE plot 저장: {TSNE_PLOT_PATH}")
    print(f"t-SNE 좌표 CSV 저장: {TSNE_CSV_PATH}")


# =========================
# 5. main
# =========================

def main():
    print("PANNs embedding 시각화 시작")
    print("-" * 70)

    df, X, labels, files, embedding_cols = load_embedding_data(INPUT_CSV)

    print(f"전체 샘플 수: {len(df)}")
    print(f"Embedding 차원: {len(embedding_cols)}")
    print("클래스별 샘플 수:")
    print(df["label"].value_counts())

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    run_tsne(X_scaled, labels, files)

    print("\n" + "-" * 70)
    print("PANNs embedding 시각화 완료")
    print(f"결과 저장 위치: {RESULT_DIR}")


if __name__ == "__main__":
    main()