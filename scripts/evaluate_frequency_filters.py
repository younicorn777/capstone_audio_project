import os
import csv
import numpy as np
import librosa
from pystoi import stoi


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
NOISE_SUMMARY_CSV = os.path.join(RESULT_DIR, "frequency_filter_noise_type_summary.csv")

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


def zero_mean(signal):
    return signal - np.mean(signal)


# =========================
# 3. 평가 지표
# =========================

def mse(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return float(np.mean((clean - estimate) ** 2))


def mae(clean, estimate):
    clean, estimate = align_length(clean, estimate)
    return float(np.mean(np.abs(clean - estimate)))


def snr_db(clean, estimate):
    clean, estimate = align_length(clean, estimate)

    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - estimate) ** 2)

    if noise_power == 0:
        return float("inf")

    return float(10 * np.log10(signal_power / noise_power))


def si_snr_db(clean, estimate, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio 계산
    """
    clean, estimate = align_length(clean, estimate)

    clean = zero_mean(clean)
    estimate = zero_mean(estimate)

    clean_energy = np.sum(clean ** 2) + eps

    target = (np.sum(estimate * clean) / clean_energy) * clean
    noise = estimate - target

    target_power = np.sum(target ** 2)
    noise_power = np.sum(noise ** 2) + eps

    return float(10 * np.log10((target_power + eps) / noise_power))


def stoi_score(clean, estimate):
    """
    STOI 계산
    - 0~1 범위
    - 1에 가까울수록 음성 명료도가 좋음
    """
    clean, estimate = align_length(clean, estimate)

    try:
        score = stoi(clean, estimate, SR, extended=False)
        return float(score)
    except Exception as e:
        print(f"[경고] STOI 계산 실패: {e}")
        return np.nan


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
# 6. 평가 row 생성
# =========================

def make_metric_row(file_name, clean_file, noise_type, method, clean, noisy_ref, estimate):
    """
    noisy_ref:
        해당 파일의 Noisy Voice 중간 구간
        improvement 계산 기준
    """
    snr_value = snr_db(clean, estimate)
    noisy_snr_value = snr_db(clean, noisy_ref)

    si_snr_value = si_snr_db(clean, estimate)
    noisy_si_snr_value = si_snr_db(clean, noisy_ref)

    stoi_value = stoi_score(clean, estimate)
    noisy_stoi_value = stoi_score(clean, noisy_ref)

    if method == "Noisy Voice":
        snr_improvement = 0.0
        si_snr_improvement = 0.0
        stoi_improvement = 0.0
    else:
        snr_improvement = snr_value - noisy_snr_value
        si_snr_improvement = si_snr_value - noisy_si_snr_value
        stoi_improvement = stoi_value - noisy_stoi_value

    return {
        "file": file_name,
        "clean_file": clean_file,
        "noise_type": noise_type,
        "method": method,

        "MSE": mse(clean, estimate),
        "MAE": mae(clean, estimate),

        "SNR_dB": snr_value,
        "SNR_Improvement_dB": snr_improvement,

        "SI_SNR_dB": si_snr_value,
        "SI_SNR_Improvement_dB": si_snr_improvement,

        "STOI": stoi_value,
        "STOI_Improvement": stoi_improvement,
    }


# =========================
# 7. 평균 요약 생성
# =========================

def summarize_by_method(results):
    methods = sorted(set(row["method"] for row in results))
    summary_rows = []

    for method in methods:
        method_rows = [row for row in results if row["method"] == method]

        summary_rows.append({
            "method": method,
            "mean_MSE": float(np.nanmean([row["MSE"] for row in method_rows])),
            "mean_MAE": float(np.nanmean([row["MAE"] for row in method_rows])),

            "mean_SNR_dB": float(np.nanmean([row["SNR_dB"] for row in method_rows])),
            "mean_SNR_Improvement_dB": float(np.nanmean([row["SNR_Improvement_dB"] for row in method_rows])),

            "mean_SI_SNR_dB": float(np.nanmean([row["SI_SNR_dB"] for row in method_rows])),
            "mean_SI_SNR_Improvement_dB": float(np.nanmean([row["SI_SNR_Improvement_dB"] for row in method_rows])),

            "mean_STOI": float(np.nanmean([row["STOI"] for row in method_rows])),
            "mean_STOI_Improvement": float(np.nanmean([row["STOI_Improvement"] for row in method_rows])),
        })

    return summary_rows


def summarize_by_noise_type_and_method(results):
    noise_types = sorted(set(row["noise_type"] for row in results))
    methods = sorted(set(row["method"] for row in results))

    summary_rows = []

    for noise_type in noise_types:
        for method in methods:
            selected_rows = [
                row for row in results
                if row["noise_type"] == noise_type and row["method"] == method
            ]

            if len(selected_rows) == 0:
                continue

            summary_rows.append({
                "noise_type": noise_type,
                "method": method,

                "mean_MSE": float(np.nanmean([row["MSE"] for row in selected_rows])),
                "mean_MAE": float(np.nanmean([row["MAE"] for row in selected_rows])),

                "mean_SNR_dB": float(np.nanmean([row["SNR_dB"] for row in selected_rows])),
                "mean_SNR_Improvement_dB": float(np.nanmean([row["SNR_Improvement_dB"] for row in selected_rows])),

                "mean_SI_SNR_dB": float(np.nanmean([row["SI_SNR_dB"] for row in selected_rows])),
                "mean_SI_SNR_Improvement_dB": float(np.nanmean([row["SI_SNR_Improvement_dB"] for row in selected_rows])),

                "mean_STOI": float(np.nanmean([row["STOI"] for row in selected_rows])),
                "mean_STOI_Improvement": float(np.nanmean([row["STOI_Improvement"] for row in selected_rows])),
            })

    return summary_rows


# =========================
# 8. CSV 저장
# =========================

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# 9. 평가 실행
# =========================

def main():
    print("주파수 영역 필터 최종 성능 평가 시작")
    print("평가 지표: MSE, MAE, SNR, SI-SNR, STOI")
    print("-" * 70)

    noisy_files = sorted([
        f for f in os.listdir(NOISY_DIR)
        if f.endswith(".wav")
    ])

    if len(noisy_files) == 0:
        print(f"[오류] 입력 폴더에 wav 파일이 없습니다: {NOISY_DIR}")
        return

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

        noise_type = get_noise_label(noisy_file)
        file_name = noisy_file.replace(".wav", "")

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

        for method_name, estimate in methods.items():
            row = make_metric_row(
                file_name=file_name,
                clean_file=clean_file,
                noise_type=noise_type,
                method=method_name,
                clean=clean,
                noisy_ref=noisy_middle,
                estimate=estimate,
            )

            results.append(row)

            print(
                f"{row['file']} | {method_name} | "
                f"MSE={row['MSE']:.6f}, "
                f"MAE={row['MAE']:.6f}, "
                f"SNRi={row['SNR_Improvement_dB']:.2f} dB, "
                f"SI-SNRi={row['SI_SNR_Improvement_dB']:.2f} dB, "
                f"STOI={row['STOI']:.4f}"
            )

    if len(results) == 0:
        print("[오류] 평가 결과가 없습니다.")
        return

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

        "SI_SNR_dB",
        "SI_SNR_Improvement_dB",

        "STOI",
        "STOI_Improvement",
    ]

    write_csv(OUTPUT_CSV, results, fieldnames)

    # method별 평균 요약 저장
    summary_rows = summarize_by_method(results)

    summary_fieldnames = [
        "method",

        "mean_MSE",
        "mean_MAE",

        "mean_SNR_dB",
        "mean_SNR_Improvement_dB",

        "mean_SI_SNR_dB",
        "mean_SI_SNR_Improvement_dB",

        "mean_STOI",
        "mean_STOI_Improvement",
    ]

    write_csv(SUMMARY_CSV, summary_rows, summary_fieldnames)

    # noise type + method별 평균 요약 저장
    noise_summary_rows = summarize_by_noise_type_and_method(results)

    noise_summary_fieldnames = [
        "noise_type",
        "method",

        "mean_MSE",
        "mean_MAE",

        "mean_SNR_dB",
        "mean_SNR_Improvement_dB",

        "mean_SI_SNR_dB",
        "mean_SI_SNR_Improvement_dB",

        "mean_STOI",
        "mean_STOI_Improvement",
    ]

    write_csv(NOISE_SUMMARY_CSV, noise_summary_rows, noise_summary_fieldnames)

    print("-" * 70)
    print("주파수 영역 필터 최종 성능 평가 완료")
    print(f"전체 결과 CSV: {OUTPUT_CSV}")
    print(f"요약 결과 CSV: {SUMMARY_CSV}")
    print(f"소음별 요약 CSV: {NOISE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()