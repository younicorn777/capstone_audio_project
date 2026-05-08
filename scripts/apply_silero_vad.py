import os
import csv
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch

from silero_vad import load_silero_vad, get_speech_timestamps


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

INPUT_DIR = os.path.join(BASE_DIR, "noisy_vad")

RESULT_DIR = os.path.join(BASE_DIR, "results", "vad")
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "vad_segments.csv")

SR = 16000


# =========================
# 2. 오디오 로드 함수
# =========================

def load_audio_librosa(path, sr=SR):
    """
    librosa로 오디오를 불러오는 함수
    - mono 변환
    - sampling rate 16 kHz 변환
    - float32 변환
    """
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def load_audio_for_silero(path, sr=SR):
    """
    Silero VAD 입력용 torch tensor 생성
    torchcodec 오류를 피하기 위해 silero_vad.read_audio()를 사용하지 않음
    """
    audio = load_audio_librosa(path, sr=sr)
    wav = torch.from_numpy(audio).float()
    return wav


def make_time_axis(audio, sr=SR):
    return np.arange(len(audio)) / sr


# =========================
# 3. non-speech 구간 계산 함수
# =========================

def get_non_speech_segments(speech_segments, duration):
    """
    speech 구간 리스트를 이용해 non-speech 구간 계산

    speech_segments:
    [
        {"start": 1.0, "end": 4.5},
        ...
    ]

    duration:
    전체 오디오 길이(sec)
    """
    non_speech = []
    current = 0.0

    for seg in speech_segments:
        start = float(seg["start"])
        end = float(seg["end"])

        if start > current:
            non_speech.append({
                "start": current,
                "end": start,
            })

        current = max(current, end)

    if current < duration:
        non_speech.append({
            "start": current,
            "end": duration,
        })

    return non_speech


# =========================
# 4. VAD 결과 시각화 함수
# =========================

def plot_vad_result(audio, speech_segments, non_speech_segments, title, output_path):
    """
    waveform 위에 speech / non-speech 구간 표시
    """
    t = make_time_axis(audio)

    plt.figure(figsize=(14, 4))
    plt.plot(t, audio, linewidth=0.8)

    # speech 구간 표시
    for i, seg in enumerate(speech_segments):
        plt.axvspan(
            seg["start"],
            seg["end"],
            alpha=0.30,
            label="speech" if i == 0 else None
        )

    # non-speech 구간 표시
    for i, seg in enumerate(non_speech_segments):
        plt.axvspan(
            seg["start"],
            seg["end"],
            alpha=0.15,
            label="non-speech" if i == 0 else None
        )

    plt.title(title)
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =========================
# 5. VAD 실행
# =========================

def main():
    print("Silero VAD 적용 시작")
    print("오디오 로딩 방식: librosa → torch tensor")
    print("-" * 60)

    model = load_silero_vad()

    wav_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".wav")
    ])

    if len(wav_files) == 0:
        print(f"[오류] 입력 폴더에 wav 파일이 없습니다: {INPUT_DIR}")
        return

    rows = []

    for wav_file in wav_files:
        input_path = os.path.join(INPUT_DIR, wav_file)

        # Silero VAD 입력용
        wav_tensor = load_audio_for_silero(input_path)

        # plotting 및 duration 계산용
        audio = load_audio_librosa(input_path)
        duration = len(audio) / SR

        # Silero VAD 실행
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=SR,
            return_seconds=True
        )

        speech_segments = [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            }
            for seg in speech_timestamps
        ]

        non_speech_segments = get_non_speech_segments(
            speech_segments=speech_segments,
            duration=duration
        )

        # CSV 저장용 row 생성
        for seg in speech_segments:
            rows.append({
                "file": wav_file,
                "segment_type": "speech",
                "start_sec": seg["start"],
                "end_sec": seg["end"],
                "duration_sec": seg["end"] - seg["start"],
            })

        for seg in non_speech_segments:
            rows.append({
                "file": wav_file,
                "segment_type": "non_speech",
                "start_sec": seg["start"],
                "end_sec": seg["end"],
                "duration_sec": seg["end"] - seg["start"],
            })

        # VAD 결과 이미지 저장
        plot_name = wav_file.replace(".wav", "_vad.png")
        plot_path = os.path.join(PLOT_DIR, plot_name)

        plot_vad_result(
            audio=audio,
            speech_segments=speech_segments,
            non_speech_segments=non_speech_segments,
            title=wav_file.replace(".wav", ""),
            output_path=plot_path
        )

        print(
            f"[완료] {wav_file} | "
            f"speech={len(speech_segments)}개, "
            f"non-speech={len(non_speech_segments)}개"
        )

    # CSV 저장
    fieldnames = [
        "file",
        "segment_type",
        "start_sec",
        "end_sec",
        "duration_sec",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 60)
    print("Silero VAD 적용 완료")
    print(f"CSV 저장 위치: {OUTPUT_CSV}")
    print(f"이미지 저장 위치: {PLOT_DIR}")


if __name__ == "__main__":
    main()