# 🛰️ Satellite Image Classification with ResNet50

![Streamlit Dashboard Preview](https://via.placeholder.com/800x400.png?text=Streamlit+Dashboard+Screenshot+Placeholder)

## 1. The Problem

**Why this matters:**
Automating land-use classification is critical for rapid urban planning, environmental conservation, and agricultural monitoring. Manually processing high-resolution Sentinel-2 satellite imagery is labor-intensive and error-prone. This project solves that by deploying an AI agent capable of identifying terrain types (e.g., Forests, Industrial Areas, Rivers) with high precision, enabling real-time geospatial analysis.

## 2. The Solution

We utilize a modern Deep Learning stack to process and classify satellite imagery.

- **Model:** Transfer Learning with **ResNet50** (ImageNet weights). The model is fine-tuned on the Sentinel-2 dataset to adapt to aerial views.
- **Accuracy:** **95%+** Validation Accuracy (after 5 epochs of fine-tuning).
- **Frontend:** A responsive **Streamlit** dashboard for easy image uploading and instant prediction.
- **Database:** PostgreSQL integration for managing image metadata and ingestion logs.

## 3. Installation & Usage

### 1. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

Launch the web interface locally:

```bash
streamlit run app/dashboard.py
```

The app will open automatically in your browser (typically at `http://localhost:8501`).

## 4. Project Structure

```text
├── app/
│   └── dashboard.py          # Main Streamlit Dashboard
├── models/
│   ├── transfer_learning.py  # Initial ResNet50 Training Script
│   └── fine_tune.py          # Fine-tuning Model Script
├── data/                     # Data Directory
│   └── raw/                  # Raw Satellite Images
├── scripts/
│   └── ingest_images.py      # Database Ingestion Script
├── requirements.txt          # Project Dependencies
└── README.md                 # Documentation
```
