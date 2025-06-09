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
- **LLM**: Local model integrated via Python
- **Data**: Custom CSV-based insurance dataset
- **Tools**: Jupyter Notebook, Pandas, Scikit-learn

---

## 📁 Project Structure

InsuranceAdvisor/
│
├── app.py # Flask app to run the backend
├── config.json # Configuration for model paths and settings
├── insurance_data.csv # Dataset used for training/evaluation
├── data_preparation.ipynb # Jupyter notebook for preprocessing data
├── evaluate_model.ipynb # Evaluation workflow
├── evaluation_report.md # Performance insights and accuracy details
├── evaluation_summary.json # Summary of evaluation metrics
├── chatbot-workflow-diagram.svg # Workflow diagram of chatbot system
├── templates/ # HTML templates for frontend
├── static/ # CSS and JS files
└── .gitignore, .gitattributes # Git metadata


**🧠 Model & Dataset**
The chatbot is trained on insurance_data.csv with intent classification and domain-specific Q&A patterns.

Data preparation and evaluation workflows are provided in Jupyter notebooks:
- data_preparation.ipynb
- evaluate_model.ipynb

**📊 Evaluation**
Performance analysis of the chatbot is documented in:
- evaluation_report.md
- evaluation_summary.json

