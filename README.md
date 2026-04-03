# ECGenius 🫀

> An AI-powered ECG analysis web application for automated arrhythmia detection and cardiac signal classification.

🌐 **Live Demo:** [ec-genius.vercel.app](https://ec-genius.vercel.app)

---

## Overview

ECGenius is a deep learning–powered web application that analyzes electrocardiogram (ECG) signals to detect and classify cardiac arrhythmias. Built with PyTorch and Flask, it provides clinicians and researchers with an intuitive interface for uploading ECG data and receiving instant AI-driven diagnostic insights.

The project combines state-of-the-art signal processing libraries with a modern web frontend to bridge the gap between raw ECG data and interpretable cardiac analysis.

---

## Features

- **Automated Arrhythmia Detection** — Deep learning model trained to classify multiple cardiac conditions from raw ECG signals
- **Signal Preprocessing** — Noise filtering, baseline wander removal, and R-peak detection powered by NeuroKit2 and BioSPPy
- **WFDB Format Support** — Compatible with PhysioNet/WFDB standard ECG file formats
- **Interactive Visualizations** — Plotly-powered charts for ECG waveform rendering and analysis results
- **REST API** — Flask backend exposing clean endpoints for ECG ingestion and prediction
- **Deployed on Vercel** — Serverless deployment for low-latency inference

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.2.1, TorchVision, TorchAudio |
| Signal Processing | NeuroKit2, BioSPPy, WFDB, SciPy |
| Data & ML | NumPy, Pandas, Scikit-learn, Imbalanced-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Backend | Flask 2.3.3, Flask-CORS |
| Monitoring | TensorBoard |
| Deployment | Vercel (Serverless Python) |

---

## Project Structure

```
ECGenesis/
├── flask_app/
│   └── app.py          # Flask application entry point & API routes
├── src/                # Model definitions, training scripts, utilities
├── logs/               # Training logs & TensorBoard outputs
├── requirements.txt    # Python dependencies
├── vercel.json         # Vercel deployment configuration
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/omroy07/ECGenesis.git
   cd ECGenesis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask development server**

   ```bash
   cd flask_app
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5000`

---

## Deployment

This project is configured for one-click deployment to [Vercel](https://vercel.com). The `vercel.json` routes all requests to the Flask app.

```bash
vercel deploy
```

---

## Dependencies

Core Python packages used:

```
torch==2.2.1
torchvision==0.17.1
torchaudio==2.2.1
wfdb==4.1.2
scipy==1.11.4
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.1
flask==2.3.3
flask-cors==4.0.0
tqdm==4.66.1
neurokit2==0.2.7
biosppy==0.8.0
imbalanced-learn==0.11.0
tensorboard==2.15.1
plotly==5.18.0
```

---

## License

This project is open source. Feel free to fork, use, and contribute.

---

## Author

**Om Roy** — [@omroy07](https://github.com/omroy07)
