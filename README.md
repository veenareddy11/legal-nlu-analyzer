# LegalBERT Contract NLU Analyzer ⚖️🤖

An AI-powered tool that automatically identifies and classifies legal clauses in contracts using a fine-tuned Legal-BERT transformer model.

## 🚀 Live Demo
Check out the interactive app here: https://huggingface.co/spaces/Veenareddyy/legal-nlu-analyzer

## 📌 Project Overview
Legal documents are long and complex. This project uses Natural Language Understanding (NLU) to help lawyers and professionals quickly categorize clauses (e.g., Termination, Governing Law, Indemnification).

### Key Features:
- **Model:** Fine-tuned `nlpaueb/legal-bert-base-uncased`.
- **Accuracy:** Achieved **80% accuracy** across **47 distinct legal categories**.
- **Dataset:** Trained on the *Legal Contract Dataset* (Kaggle).
- **Interface:** Built with **Gradio** for real-time inference.

## 🛠️ Tech Stack
- **Language:** Python
- **Frameworks:** PyTorch, Hugging Face Transformers
- **Deployment:** Hugging Face Spaces
- **Tools:** Google Colab (GPU-accelerated training)

## 📊 Results
The model demonstrates high confidence (95%+) in identifying core clauses like "Termination for Convenience" and "Governing Law."

## 📂 Repository Structure
- `app.py`: The Gradio interface script.
- `requirements.txt`: Python dependencies.
- `Legal_NLU_Notebook.ipynb`: The full training and evaluation pipeline.
