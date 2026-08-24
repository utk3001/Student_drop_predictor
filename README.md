# FairPredict: Student Outcome & Dropout Prediction API

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-API-green.svg)
![React](https://img.shields.io/badge/React-Frontend-blue.svg)
![Machine Learning](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Fairness](https://img.shields.io/badge/Algorithmic-Fairness-purple.svg)
![XAI](https://img.shields.io/badge/Explainable-AI-red.svg)

## 📌 Description

**FairPredict** is a machine learning-powered web application designed to predict student dropout rates and academic outcomes. Developed as a course project for **CS698Y: Human AI Interaction**, this project places a heavy emphasis on **Algorithmic Fairness** and **Explainable AI (XAI)**.

Rather than relying on a black-box model, FairPredict actively mitigates bias by neutralizing protected attributes (such as gender, age, and nationality) and utilizes LIME and SHAP to provide transparent, human-readable justifications for every prediction it makes. The project consists of a Python/Flask machine learning backend and a React/Tailwind CSS frontend dashboard, and is currently deployed on Render.

## 🎯 Key ML Features

- **High Accuracy Inference:** Achieves an 88.1% overall prediction accuracy on real-world education data using `scikit-learn`.
- **Bias Mitigation:** Implements algorithmic fairness techniques to ensure predictions are not skewed by demographic data, improving overall fairness metrics by 50-60%.
- **Explainable AI (XAI):** 
  - **Local Explanations:** Uses **LIME** (Local Interpretable Model-agnostic Explanations) to generate localized feature justifications for individual student predictions.
  - **Global Explanations:** Uses **SHAP** (SHapley Additive exPlanations) to provide global feature importance metrics and visualizations.
- **RESTful ML Serving:** A robust Flask API that serves model predictions and interfaces with a MongoDB database for student records.

## 🛠 Tech Stack

- **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `LIME`, `SHAP`
- **Backend API:** Python, Flask, PyMongo
- **Database:** MongoDB
- **Frontend:** React.js, Tailwind CSS, Axios
- **Deployment:** Render

## 🏷️ GitHub Topics

When publishing this repository to GitHub, consider adding the following tags to the "Topics" section to increase visibility among the Data Science and ML communities:

`machine-learning` `explainable-ai` `algorithmic-fairness` `xai` `shap` `lime` `student-dropout-prediction` `flask-api` `react` `scikit-learn` `data-science`

## 🚀 Getting Started

### Prerequisites
- Node.js (v14+)
- Python (3.8+)
- MongoDB connection string

### Backend (Machine Learning API)
1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with your MongoDB URL:
   ```env
   MONGO_URL="your_mongodb_connection_string"
   ```
4. Run the Flask server:
   ```bash
   python app.py
   ```
   *The API will start on http://localhost:8000*

### Frontend (React Dashboard)
1. Open a new terminal and navigate to the client directory:
   ```bash
   cd client
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
   *The frontend will start on http://localhost:3000*

## 📁 Project Structure

```text
.
├── client/                 # React frontend application
│   ├── public/             
│   ├── src/                
│   └── package.json        
├── server/                 # Flask API and Machine Learning models
│   ├── app.py              # Main Flask application entry point
│   ├── predict_student.py  # Inference logic, LIME/SHAP explanations, fairness metrics
│   ├── model_server.py     # Evaluation and metrics logic
│   ├── *.pkl               # Serialized ML models and scalers
│   └── requirements.txt    # Python dependencies
├── proposed_improvements.md# Roadmap for future MLOps & Advanced ML features
└── README.md               # Project documentation
```

## 🔮 Future Work (MLOps Roadmap)
Check out [`proposed_improvements.md`](./proposed_improvements.md) for the planned roadmap, which includes migrating to FastAPI, implementing MLflow for experiment tracking, and utilizing advanced ensemble methods.
