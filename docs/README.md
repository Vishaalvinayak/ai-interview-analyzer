# AI Interview Answer Analyzer

ML-based system to evaluate interview answers and provide feedback.

## Current Status
- ✅ Project structure created
- ⏳ Dataset creation in progress
- ⏳ Model training pending
- ⏳ API development pending
- ⏳ Frontend development pending

## Setup Instructions

### 1. Create Dataset
Currently: 12 samples in `dataset/dataset.json`
**Target: 200-300 samples minimum**

### 2. Install Python Dependencies
```bash
cd ml_training
pip install -r requirements.txt
```

### 3. Train Model
```bash
python train_model.py
```

## Project Structure
```
interview-analyzer/
├── dataset/
│   └── dataset.json         # Training data (12/300 complete)
├── ml_training/
│   ├── requirements.txt     # Python packages
│   └── train_model.py       # Training script
├── ml_service/
│   ├── requirements.txt     # API packages
│   └── ml_api.py           # FastAPI service (coming soon)
└── docs/                    # Documentation
```

## Tech Stack
- **ML**: DistilBERT (Hugging Face Transformers)
- **Backend**: FastAPI + Python
- **Frontend**: Next.js + TypeScript (coming soon)

## Author
[Your Name]
Final Year Project - 2024