# Audio Noise Enhancement & Noise Classification System

## 📌 Project Overview

본 프로젝트는 소음이 포함된 음성 신호를 입력받아 화자 음성을 정제하고, 동시에 주변 소음 환경을 분류하는 통합 음성 분석 시스템이다.

기존의 음성 향상(Speech Enhancement) 연구가 음성 품질 개선에 집중하는 것과 달리, 본 프로젝트는 음성 정제와 소음 환경 분석을 하나의 시스템으로 통합하여 사용자에게 더욱 풍부한 음향 정보를 제공하는 것을 목표로 한다.

---

## 🎯 Project Objectives

- Wiener Filter를 이용한 음성 신호 정제
- PANNs 기반 음향 특징 추출
- XGBoost 기반 소음 환경 분류
- 음성 신호의 시각적 분석 지원
- Streamlit 기반 사용자 인터페이스 구현

---

## 🚀 Main Features

### 1. Speech Enhancement

- Wiener Filter 기반 음성 정제
- 배경 소음 감소
- 화자 음성 명료도 향상

### 2. Noise Classification

- PANNs Embedding 추출
- XGBoost 분류기 적용
- 소음 환경 자동 분류

지원 클래스:

- Air Conditioner
- Engine Idling
- Dog Bark

### 3. Audio Visualization

- Original Waveform
- Enhanced Waveform
- Original Spectrogram
- Enhanced Spectrogram

### 4. Interactive Demo

- 음성 파일 업로드
- 원본 음성 재생
- 정제 음성 재생
- 예측 결과 확인

---

## 🏗️ System Architecture

```text
Input Audio (.wav)
        │
        ▼
 ┌─────────────┐
 │Wiener Filter│
 └─────────────┘
        │
        ▼
 Enhanced Audio


Input Audio (.wav)
        │
        ▼
 ┌─────────────┐
 │    PANNs    │
 └─────────────┘
        │
        ▼
 Embedding Vector
        │
        ▼
 ┌─────────────┐
 │   XGBoost   │
 └─────────────┘
        │
        ▼
 Noise Class
```

---

## 📂 Dataset

### Clean Speech

- Mozilla Common Voice

### Noise Dataset

- UrbanSound8K

사용 클래스:

| Class | Description |
|---------|---------|
| Air Conditioner | 에어컨 소음 |
| Engine Idling | 엔진 공회전 소음 |
| Dog Bark | 개 짖는 소리 |

---

## 🧠 Technologies

### Audio Processing

- Librosa
- Noisereduce
- NumPy
- SciPy

### Deep Learning

- PyTorch
- PANNs

### Machine Learning

- Scikit-learn
- XGBoost

### Visualization

- Matplotlib
- Seaborn

### Web Application

- Streamlit

---

## 📁 Project Structure

```text
.
├── app.py
├── README.md
├── capstone_week9_dataset/ (gitignored - large dataset, not pushed to repo)
│   ├── clean/
│   ├── external_noise_dataset_verified/
│   │   ├── air/
│   │   ├── dog/
│   │   └── engine/
│   ├── UrbanSound8K/
│   │   ├── audio/
│   │   └── metadata/
│   ├── yamnet_candidate_ranking/
│   ├── yamnet_filtered_candidates/
│   └── results/
│       ├── demo_audio/
│       ├── evaluation/
│       └── plots/
├── models/
│   ├── panns/
│   └── xgboost/
├── scripts/
│   ├── apply_kalman_filter.py
│   ├── apply_moving_average.py
│   └── evaluate_filters.py
├── temp_streamlit/
└── results/ (gitignored - generated outputs, not pushed)
```

---

## 👥 Team

- 윤영현
- 장병관
- 박선우
- 윤순천
- 양푸른바다

Department of AI Data Engineering

---

## 📖 Project Description

본 프로젝트는 음성 정제와 소음 분류를 결합한 통합 음성 분석 시스템을 제안한다. Wiener Filter를 통해 화자 음성을 정제하고, PANNs 기반 특징 추출과 XGBoost 분류기를 활용하여 주변 소음 환경을 분류한다. 또한 Streamlit 기반 인터페이스를 통해 사용자가 직관적으로 결과를 확인할 수 있도록 설계하였다.