# 📧 Spam Message Detection Web App

This project is a Machine Learning based Spam Message Detector built using:

- Python
- Flask
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

## 🚀 Features
✔ Detects whether a message is Spam or Not Spam  
✔ Web interface to test messages  
✔ API endpoint support  
✔ Text preprocessing using NLP  

## 🧠 Model Used
Logistic Regression trained on SMS Spam dataset.

## ▶ How to Run

1. Install requirements:
   pip install -r requirements.txt

2. Run app:
   python spam_massage.py

3. Open browser:
   http://127.0.0.1:5000/form

## 📡 API Usage

POST request to:
/predict

Body:
{
  "message": "Free offer just for you"
}

## 📌 Author
Your Name

