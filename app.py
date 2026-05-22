import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from panns_inference import AudioTagging


# =========================
# 1. 기본 설정
# =========================

BASE_DIR = "capstone_week9_dataset"

MODEL_DIR = os.path.join(BASE_DIR, "models")
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "panns_xgboost.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

PANNs_CHECKPOINT_PATH = os.path.join(
    os.path.expanduser("~"),
    "panns_data",
    "Cnn14_mAP=0.431.pth"
)

TEMP_DIR = "temp_streamlit"
os.makedirs(TEMP_DIR, exist_ok=True)

AUDIO_SR = 16000
PANNS_SR = 32000


# =========================
# 2. Streamlit 설정
# =========================

st.set_page_config(
    page_title="Intelligent Voice Analysis System",
    layout="wide"
)

st.title("🎧 지능형 음성 분석 시스템")
st.info(
    "본 시스템은 noisy voice를 입력받아 Wiener Filter로 화자 음성을 정제하고, "
    "PANNs embedding + XGBoost 모델을 이용해 주변 소음 환경을 분류합니다."
)


# =========================
# 3. 캐시 로딩
# =========================

@st.cache_resource
def load_xgboost_model():
    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError(f"XGBoost 모델 파일이 없습니다: {XGBOOST_MODEL_PATH}")

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder 파일이 없습니다: {LABEL_ENCODER_PATH}")

    model = joblib.load(XGBOOST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return model, label_encoder


@st.cache_resource
def load_panns_model():
    if not os.path.exists(PANNs_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"PANNs checkpoint 파일이 없습니다: {PANNs_CHECKPOINT_PATH}\n"
            "먼저 scripts/setup_panns_files.py를 실행하세요."
        )

    model = AudioTagging(
        checkpoint_path=PANNs_CHECKPOINT_PATH,
        device="cpu"
    )

    return model


# =========================
# 4. 오디오 유틸 함수
# =========================

def load_audio(path, sr=AUDIO_SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)
    return audio


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    return audio / max_val


def save_audio(path, audio, sr=AUDIO_SR):
    sf.write(path, audio, sr)


# =========================
# 5. Wiener Filter
# =========================

def estimate_noise_spectrum(audio, sr=AUDIO_SR, noise_sec=0.5, n_fft=1024, hop_length=256):
    """
    입력 음성의 앞부분 noise_sec 구간을 noise-only 구간으로 가정하여
    noise power spectrum을 추정한다.
    """
    noise_len = int(noise_sec * sr)

    if len(audio) < noise_len:
        noise_audio = audio
    else:
        noise_audio = audio[:noise_len]

    noise_stft = librosa.stft(
        noise_audio,
        n_fft=n_fft,
        hop_length=hop_length
    )

    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

    return noise_power


def apply_wiener_filter(
    audio,
    sr=AUDIO_SR,
    noise_sec=0.5,
    n_fft=1024,
    hop_length=256,
    gain_floor=0.05
):
    """
    간단한 frequency-domain Wiener Filter.
    앞부분 noise-only 구간에서 noise spectrum을 추정한 뒤,
    전체 noisy voice에 Wiener gain을 적용한다.
    """
    audio = audio.astype(np.float32)

    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length
    )

    power = np.abs(stft) ** 2
    phase = np.angle(stft)

    noise_power = estimate_noise_spectrum(
        audio=audio,
        sr=sr,
        noise_sec=noise_sec,
        n_fft=n_fft,
        hop_length=hop_length
    )

    signal_power_est = np.maximum(power - noise_power, 0.0)

    gain = signal_power_est / (signal_power_est + noise_power + 1e-10)
    gain = np.maximum(gain, gain_floor)

    enhanced_stft = gain * np.abs(stft) * np.exp(1j * phase)

    enhanced_audio = librosa.istft(
        enhanced_stft,
        hop_length=hop_length,
        length=len(audio)
    )

    enhanced_audio = normalize_audio(enhanced_audio)

    return enhanced_audio.astype(np.float32)


# =========================
# 6. 시각화 함수
# =========================

def plot_waveform(audio, sr, title):
    fig, ax = plt.subplots(figsize=(10, 3))

    time = np.arange(len(audio)) / sr

    ax.plot(time, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


def plot_spectrogram(audio, sr, title):
    fig, ax = plt.subplots(figsize=(10, 3))

    spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)),
        ref=np.max
    )

    img = librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    st.pyplot(fig)
    plt.close(fig)


# =========================
# 7. PANNs embedding + XGBoost 분류
# =========================

def load_audio_for_panns(path, sr=PANNS_SR):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)

    min_len = sr

    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    return audio[None, :]


def extract_panns_embedding(panns_model, audio_batch):
    clipwise_output, embedding = panns_model.inference(audio_batch)
    return embedding


def predict_noise_environment(audio_path):
    xgb_model, label_encoder = load_xgboost_model()
    panns_model = load_panns_model()

    audio_batch = load_audio_for_panns(audio_path)
    embedding = extract_panns_embedding(panns_model, audio_batch)

    pred_encoded = xgb_model.predict(embedding)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    probabilities = None

    if hasattr(xgb_model, "predict_proba"):
        proba = xgb_model.predict_proba(embedding)[0]
        class_labels = label_encoder.inverse_transform(np.arange(len(proba)))
        probabilities = {
            label: float(prob)
            for label, prob in zip(class_labels, proba)
        }

    return pred_label, probabilities


# =========================
# 8. Sidebar
# =========================

st.sidebar.header("⚙️ 설정")

noise_sec = st.sidebar.slider(
    "Noise profile 추정 구간 (초)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1
)

gain_floor = st.sidebar.slider(
    "Wiener gain floor",
    min_value=0.00,
    max_value=0.30,
    value=0.05,
    step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("**음성 정제 모델**")
st.sidebar.write("Wiener Filter")

st.sidebar.markdown("**소음 분류 모델**")
st.sidebar.write("PANNs embedding + XGBoost")


# =========================
# 9. Main UI
# =========================

uploaded_file = st.file_uploader(
    "noisy voice wav 파일을 업로드하세요.",
    type=["wav"]
)

if uploaded_file is None:
    st.info("왼쪽 또는 위의 업로드 영역에서 wav 파일을 업로드하면 분석이 시작됩니다.")
    st.stop()


# 업로드 파일 저장
input_path = os.path.join(TEMP_DIR, uploaded_file.name)

with open(input_path, "wb") as f:
    f.write(uploaded_file.getbuffer())


# 오디오 로드
original_audio = load_audio(input_path, sr=AUDIO_SR)
original_audio = normalize_audio(original_audio)

st.subheader("1. 원본 음성")

st.audio(input_path)

col1, col2 = st.columns(2)

with col1:
    plot_waveform(original_audio, AUDIO_SR, "Original Waveform")

with col2:
    plot_spectrogram(original_audio, AUDIO_SR, "Original Spectrogram")


# =========================
# 10. Wiener Filter 적용
# =========================

st.subheader("2. Wiener Filter 기반 음성 정제")

with st.spinner("Wiener Filter 적용 중..."):
    enhanced_audio = apply_wiener_filter(
        audio=original_audio,
        sr=AUDIO_SR,
        noise_sec=noise_sec,
        gain_floor=gain_floor
    )

enhanced_path = os.path.join(TEMP_DIR, "enhanced_wiener.wav")
save_audio(enhanced_path, enhanced_audio, AUDIO_SR)

st.audio(enhanced_path)

col3, col4 = st.columns(2)

with col3:
    plot_waveform(enhanced_audio, AUDIO_SR, "Enhanced Waveform")

with col4:
    plot_spectrogram(enhanced_audio, AUDIO_SR, "Enhanced Spectrogram")


# =========================
# 11. 소음 환경 분류
# =========================

st.subheader("3. 주변 소음 환경 분류")

with st.spinner("PANNs embedding 추출 및 XGBoost 분류 중..."):
    predicted_label, probabilities = predict_noise_environment(input_path)

label_display = {
    "air": "air_conditioner",
    "dog": "dog_bark",
    "engine": "engine / vehicle"
}

st.success(f"예측된 소음 환경: **{label_display.get(predicted_label, predicted_label)}**")

noise_description = {
    "air": "실내 공조/에어컨 계열 소음으로 추정됩니다.",
    "dog": "개 짖는 소리 계열 소음으로 추정됩니다.",
    "engine": "엔진/차량 계열 소음으로 추정됩니다."
}

st.write(noise_description.get(predicted_label, "해당 소음 환경으로 추정됩니다."))

if probabilities is not None:
    st.markdown("### Class Probability")

    for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        display_name = label_display.get(label, label)
        st.write(f"{display_name}: {prob:.4f}")
        st.progress(float(prob))


# =========================
# 12. 최종 요약
# =========================

st.subheader("4. 분석 요약")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric("음성 정제", "Wiener Filter")

with summary_col2:
    st.metric("소음 분류", "PANNs + XGBoost")

with summary_col3:
    st.metric("예측 결과", label_display.get(predicted_label, predicted_label))

st.markdown("---")

st.caption(
    "본 시스템은 noisy voice를 입력받아 Wiener Filter로 화자 음성을 정제하고, "
    "PANNs embedding과 XGBoost를 이용해 주변 소음 환경을 분류합니다."
)