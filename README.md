# 🛡️ Insurance Advisor Chatbot

An intelligent, locally hosted chatbot for providing insurance-related assistance to users. This project leverages a local LLM (Language Model), with a Flask backend and a clean HTML/CSS/JS-based frontend. It aims to automate insurance query handling and improve customer experience in the insurance domain.

---

## 🚀 Features

- Conversational interface for insurance queries
- Trained on domain-specific data (insurance dataset)
- Flask-based backend with LLM integration
- Interactive frontend using HTML/CSS/JavaScript
- Evaluation and performance reporting
- Modular design for easy enhancements

---

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask)
- **LLM**: Local model integrated via Python (DialoGPT-small + LoRA/PEFT)
- **Data**: Custom CSV-based insurance dataset
- **Tools**: Jupyter Notebook, Pandas, Scikit-learn, HuggingFace Transformers, Datasets

---

## 📁 Project Structure

```
InsuranceAdvisor/
│
├── app.py                      # Flask app to run the backend
├── config.json                 # Configuration for model paths and settings
├── insurance_data.csv          # Dataset used for training/evaluation
├── data_preparation.ipynb      # Jupyter notebook for preprocessing data
├── model_training.ipynb        # Notebook for model training (LoRA/PEFT)
├── model_testing.ipynb         # Notebook for model testing and analysis
├── evaluate_model.ipynb        # Evaluation workflow
├── evaluation_report.md        # Performance insights and accuracy details
├── evaluation_summary.json     # Summary of evaluation metrics
├── model_evaluation_results.csv# Detailed evaluation results
├── train_data.json             # Training data (JSON)
├── val_data.json               # Validation data (JSON)
├── static/                     # CSS and JS files
├── templates/                  # HTML templates for frontend
├── insurance-chatbot-cpu/      # Saved model and tokenizer
├── insurance-chatbot-model-cpu/# Model checkpoints
└── .gitignore, .gitattributes  # Git metadata
```

---

## 🧠 Model & Dataset

- The chatbot is trained on `insurance_data.csv` with intent classification and domain-specific Q&A patterns.
- Data preparation and evaluation workflows are provided in Jupyter notebooks:
  - [`data_preparation.ipynb`](data_preparation.ipynb)
  - [`evaluate_model.ipynb`](evaluate_model.ipynb)
- Model training uses HuggingFace Transformers with LoRA/PEFT for efficient fine-tuning on CPU.

---

## 📊 Evaluation

Performance analysis of the chatbot is documented in:
- [`evaluation_report.md`](evaluation_report.md)
- [`evaluation_summary.json`](evaluation_summary.json)

---

## ⚡ Quickstart

1. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Run `data_preparation.ipynb` to generate `train_data.json` and `val_data.json`.

3. **Train the model**  
   Run `model_training.ipynb` to fine-tune the chatbot model.

4. **Evaluate the model**  
   Run `model_testing.ipynb` or `evaluate_model.ipynb` for performance metrics.

5. **Start the server**  
   ```
   python app.py
   ```
   Visit [http://localhost:5000](http://localhost:5000) in your browser.


---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT/LoRA](https://github.com/huggingface/peft)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)


