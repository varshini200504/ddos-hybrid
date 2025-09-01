# Hybrid AI-Powered DDoS Detection System

## Overview
A real-time, hybrid approach combining Graph Neural Networks (GNN) and NLP-based semantic analysis for DDoS attack detection and mitigation.

## Tech Stack
- Python 3.11, TensorFlow/Keras, Spektral, Transformers
- FastAPI for inference
- GCP (Vertex AI, GCS, Compute Engine)

## Structure
- `data/` – datasets & processed features
- `src/` – core source code
- `notebooks/` – experiments & EDA
- `infra/` – deployment scripts

## Installation
```bash
git clone https://github.com/varshini200504/ddos-hybrid.git
cd ddos-hybrid
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
