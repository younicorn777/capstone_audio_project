import os
import csv
import numpy as np
import librosa


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy_vad")

SS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "spectral_subtraction_grid_search",
    "best_audio"
)

WF_DIR = os.path.join(
    BASE_DIR,
    "results",
    "wiener_filter_grid_search",
    "best_audio"
)

RESULT_DIR = os.path.join(BASE_DIR, "results", "final_filter_evaluation")
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "frequency_filter_evaluation_results.csv")
SUMMARY_CSV = os.path.join(RESULT_DIR, "frequency_filter_summary_results.csv")

SR = 16000
FRONT_NOISE_ONLY_SEC = 1.0


# =========================
# 2. 오디오 로드
# =========================

def load_audio(path, sr=SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def align_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


# =========================
# 3. 평가 지표
# =========================

def mse(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return np.mean((clean - estimate) ** 2)


def mae(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return np.mean(np.abs(clean - estimate))


def snr_db(clean, estimate):
    clean, estimate = align_length(clean, estimate)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - estimate) ** 2)

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


def snr_improvement(clean, noisy, estimate):
    before = snr_db(clean, noisy)
    after = snr_db(clean, estimate)
    return after - before


# =========================
# 4. 파일 이름 매칭
# =========================

def get_clean_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> clean_01.wav
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    clean_number = parts[2].replace("clean", "")
    return f"clean_{clean_number}.wav"


def get_ss_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> ss_best_clean01_air.wav
    """
    return noisy_file.replace("noisy_vad_", "ss_best_")


def get_wf_file_from_noisy_vad(noisy_file):
    """
    noisy_vad_clean01_air.wav -> wf_best_clean01_air.wav
    """
    return noisy_file.replace("noisy_vad_", "wf_best_")


def get_noise_label(noisy_file):
    """
    noisy_vad_clean01_air.wav -> air
    """
    base = noisy_file.replace(".wav", "")
    parts = base.split("_")
    return parts[-1]


# =========================
# 5. 중간 clean+noise 구간 추출
# =========================

def extract_middle_region(audio, clean_len):
    """
    앞 1초 noise-only 구간을 제외하고,
    clean voice와 대응되는 중간 구간만 추출
    """
    front_len = int(FRONT_NOISE_ONLY_SEC * SR)
    return audio[front_len:front_len + clean_len]


# =========================
# 6. 평가 실행
# =========================

def main():
    print("주파수 영역 필터 최종 성능 평가 시작")
    print("-" * 60)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    results = []

    for noisy_file in noisy_files:
        clean_file = get_clean_file_from_noisy_vad(noisy_file)
        ss_file = get_ss_file_from_noisy_vad(noisy_file)
        wf_file = get_wf_file_from_noisy_vad(noisy_file)

        clean_path = os.path.join(CLEAN_DIR, clean_file)
        noisy_path = os.path.join(NOISY_DIR, noisy_file)
        ss_path = os.path.join(SS_DIR, ss_file)
        wf_path = os.path.join(WF_DIR, wf_file)

        if not os.path.exists(clean_path):
            print(f"[경고] Clean 파일 없음: {clean_path}")
            continue

        if not os.path.exists(noisy_path):
            print(f"[경고] Noisy 파일 없음: {noisy_path}")
            continue

        clean = load_audio(clean_path)
        noisy_full = load_audio(noisy_path)

        clean_len = len(clean)

        noisy_middle = extract_middle_region(noisy_full, clean_len)

        methods = {
            "Noisy Voice": noisy_middle,
        }

        if os.path.exists(ss_path):
            ss_full = load_audio(ss_path)
            methods["Spectral Subtraction"] = extract_middle_region(ss_full, clean_len)
        else:
            print(f"[경고] Spectral Subtraction 파일 없음: {ss_path}")

        if os.path.exists(wf_path):
            wf_full = load_audio(wf_path)
            methods["Wiener Filter"] = extract_middle_region(wf_full, clean_len)
        else:
            print(f"[경고] Wiener Filter 파일 없음: {wf_path}")

        noise_label = get_noise_label(noisy_file)

        for method_name, estimate in methods.items():
            row = {
                "file": noisy_file.replace(".wav", ""),
                "clean_file": clean_file,
                "noise_type": noise_label,
                "method": method_name,
                "MSE": mse(clean, estimate),
                "MAE": mae(clean, estimate),
                "SNR_dB": snr_db(clean, estimate),
                "SNR_Improvement_dB": (
                    0.0 if method_name == "Noisy Voice"
                    else snr_improvement(clean, noisy_middle, estimate)
                ),
            }

            results.append(row)

            print(
                f"{row['file']} | {method_name} | "
                f"MSE={row['MSE']:.6f}, "
                f"MAE={row['MAE']:.6f}, "
                f"SNR={row['SNR_dB']:.2f} dB, "
                f"Improvement={row['SNR_Improvement_dB']:.2f} dB"
            )

    # 전체 결과 저장
    fieldnames = [
        "file",
        "clean_file",
        "noise_type",
        "method",
        "MSE",
        "MAE",
        "SNR_dB",
        "SNR_Improvement_dB",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # method별 평균 요약
    methods = sorted(set(row["method"] for row in results))
    summary_rows = []

    for method in methods:
        method_rows = [row for row in results if row["method"] == method]

        summary_rows.append({
            "method": method,
            "mean_MSE": np.mean([row["MSE"] for row in method_rows]),
            "mean_MAE": np.mean([row["MAE"] for row in method_rows]),
            "mean_SNR_dB": np.mean([row["SNR_dB"] for row in method_rows]),
            "mean_SNR_Improvement_dB": np.mean([row["SNR_Improvement_dB"] for row in method_rows]),
        })

    summary_fieldnames = [
        "method",
        "mean_MSE",
        "mean_MAE",
        "mean_SNR_dB",
        "mean_SNR_Improvement_dB",
    ]

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 60)
    print("주파수 영역 필터 최종 성능 평가 완료")
    print(f"전체 결과 CSV: {OUTPUT_CSV}")
    print(f"요약 결과 CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()