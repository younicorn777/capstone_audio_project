import os
import pandas as pd


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

YAMNET_PRED_CSV = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "yamnet_direct",
    "yamnet_direct_predictions.csv"
)

PANNS_PRED_CSV = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "panns_direct",
    "panns_direct_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ml",
    "direct_score_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. 분석 함수
# =========================

def analyze_direct_scores(model_name, csv_path):
    print("\n" + "=" * 80)
    print(f"{model_name} Direct score 분석")
    print("=" * 80)

    if not os.path.exists(csv_path):
        print(f"[오류] 파일이 없습니다: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    required_cols = {
        "file",
        "true_label",
        "predicted_label",
        "air_score",
        "dog_score",
        "engine_score"
    }

    if not required_cols.issubset(df.columns):
        print("[오류] 필요한 컬럼이 없습니다.")
        print(f"필요 컬럼: {required_cols}")
        print(f"현재 컬럼: {df.columns.tolist()}")
        return None

    # score margin 계산
    df["engine_minus_air"] = df["engine_score"] - df["air_score"]
    df["air_minus_engine"] = df["air_score"] - df["engine_score"]

    # 정답 여부
    df["is_correct"] = df["true_label"] == df["predicted_label"]

    # 전체 요약
    total = len(df)
    correct = df["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0

    print(f"\n전체 sample 수: {total}")
    print(f"정답 수: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    # 클래스별 예측 분포
    print("\n[클래스별 예측 분포]")
    pred_table = pd.crosstab(
        df["true_label"],
        df["predicted_label"],
        rownames=["Actual"],
        colnames=["Predicted"]
    )
    print(pred_table)

    # 클래스별 score 평균
    print("\n[true_label별 평균 score]")
    score_summary = df.groupby("true_label")[
        ["air_score", "dog_score", "engine_score", "engine_minus_air"]
    ].mean()

    print(score_summary)

    # air 샘플만 분석
    air_df = df[df["true_label"] == "air"].copy()

    print("\n[air sample 분석]")
    print(f"air sample 수: {len(air_df)}")

    if len(air_df) > 0:
        air_correct = (air_df["predicted_label"] == "air").sum()
        air_to_engine = (air_df["predicted_label"] == "engine").sum()
        air_to_dog = (air_df["predicted_label"] == "dog").sum()

        print(f"air → air: {air_correct}")
        print(f"air → engine: {air_to_engine}")
        print(f"air → dog: {air_to_dog}")

        print("\n[air sample 평균 score]")
        print(air_df[["air_score", "dog_score", "engine_score", "engine_minus_air"]].mean())

        # air인데 engine으로 오분류된 파일
        air_engine_error = air_df[air_df["predicted_label"] == "engine"].copy()
        air_engine_error = air_engine_error.sort_values(
            by="engine_minus_air",
            ascending=False
        )

        error_csv = os.path.join(
            OUTPUT_DIR,
            f"{model_name.lower()}_air_to_engine_errors.csv"
        )

        air_engine_error.to_csv(error_csv, index=False, encoding="utf-8-sig")

        print(f"\nair → engine 오분류 파일 수: {len(air_engine_error)}")
        print(f"오분류 상세 저장: {error_csv}")

        if len(air_engine_error) > 0:
            print("\n[air → engine 오분류 상위 10개]")
            print(
                air_engine_error[
                    [
                        "file",
                        "true_label",
                        "predicted_label",
                        "air_score",
                        "engine_score",
                        "engine_minus_air"
                    ]
                ].head(10)
            )

    # 전체 분석 결과 저장
    output_csv = os.path.join(
        OUTPUT_DIR,
        f"{model_name.lower()}_direct_score_analysis.csv"
    )

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n전체 분석 결과 저장: {output_csv}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "score_summary": score_summary,
        "prediction_table": pred_table,
    }


# =========================
# 3. main
# =========================

def main():
    print("Direct prediction score 기반 오분류 원인 분석 시작")

    yamnet_result = analyze_direct_scores(
        model_name="YAMNet",
        csv_path=YAMNET_PRED_CSV
    )

    panns_result = analyze_direct_scores(
        model_name="PANNs",
        csv_path=PANNS_PRED_CSV
    )

    print("\n" + "=" * 80)
    print("분석 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()