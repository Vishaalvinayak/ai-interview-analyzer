 🎯 AI Interview Answer Analyzer



Machine learning system that evaluates interview answers using fine-tuned DistilBERT transformer model.



📊 Performance



Test Accuracy: 96.15%

Model: DistilBERT (66M parameters)

Dataset: 126 manually curated samples

Response Time:~300ms on CPU



✨ Features



 ✅ Real-time classification (weak/average/strong)

 ✅ Numerical scoring (0-100 scale)

 ✅ Domain-specific keyword analysis

 ✅ Weak language pattern detection

 ✅ Actionable improvement suggestions

 ✅ Modern web interface



 🏗️ Tech Stack



Backend:

 Python 3.13

 PyTorch 2.10.0

Transformers 5.1.0 (Hugging Face)

FastAPI

 DistilBERT



Frontend:

 Next.js 14

 React 18

 TypeScript

 Tailwind CSS



 🚀 Quick Start



 Prerequisites

 Python 3.10+

 Node.js 20.x LTS

 8GB RAM minimum



 Backend Setup

```bash

cd ml\_service

pip install -r requirements.txt



Train the model (8 minutes)

cd ../ml\_training

python train\_model.py



 Start API server

cd ../ml\_service

python -m uvicorn ml\_api:app --reload --port 8000

```



 Frontend Setup

```bash

cd frontend

npm install

npm run dev

```



Access Application: http://localhost:3000



 📈 Model Performance



Classification Report



| Class | Precision | Recall | F1-Score | Support |

|-------|-----------|--------|----------|---------|

| Weak | 0.90 | 1.00 | 0.95 | 9 |

| Average | 1.00 | 0.89 | 0.94 | 9 |

| Strong | 1.00 | 1.00 | 1.00 | 8 |



Overall Accuracy: 96.15% (25/26 correct predictions)



Confusion Matrix

```

&#x20;             Predicted

&#x20;          Weak  Avg  Strong

Actual:

Weak          9    0       0

Average       1    8       0

Strong        0    0       8

```



Only 1 error: 1 average answer misclassified as weak



 📁 Project Structure

```

interview-analyzer/

├── dataset/

│   └── dataset.json          # 126 labeled interview answers

├── ml\_training/

│   ├── train\_model.py        # Model training script

│   └── requirements.txt

├── ml\_service/

│   ├── ml\_api.py            # FastAPI backend

│   ├── requirements.txt

│   └── model\_saved/         # Trained model (not in repo - too large)

└── frontend/

&#x20;   ├── app/

&#x20;   │   └── page.tsx         # Main React component

&#x20;   ├── package.json

&#x20;   └── tailwind.config.ts

```



 🎓 Academic Project



Final Year B.Tech Computer Science Project - 2024



\*\*Comparison with Existing Systems:\*\*



| Approach | Accuracy | Our Advantage |

|----------|----------|---------------|

| Keyword Matching | 45-55% | +41-51% |

| Random Forest | 72% | +24% |

| BiLSTM | 76% | +20% |

| Our DistilBERT | 96.15%| Baseline |



📄 License



MIT License



 👤 Author



Vishaal Vinayak







Note: The trained model files are not included due to size (268MB). Run `python train\_model.py` to train the model locally.

