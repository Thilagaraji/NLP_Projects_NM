# 🤖 NLP_PROJECTS_NM

Welcome to **NLP_PROJECTS_NM**, a collection of practical NLP projects designed for learning and experimentation.

This repository includes:

1. 💬 **Customer Support Chatbot** – An AI-driven customer support chatbot using a pre-trained transformer model.
2. 📧 **Spam Email Detection** – A spam detection system built using Naive Bayes classification on SMS messages.

---
## 📁 Project Structure

NLP_PROJECTS_NM/
├── chatbot.py            # Customer support chatbot (Flask + DialoGPT)
├── spam_detector.py      # Spam email detector (ML + Naive Bayes)
├── spam.csv              # Dataset for spam classification (user-supplied)
└── README.md             # Project documentation

## 💬 Project 1: Customer Support Chatbot

### 🧾 Description

This chatbot uses Hugging Face’s `DialoGPT` to simulate human-like conversations for customer support purposes. The chatbot API is built with Flask, and it generates responses to user queries.

### 🛠 Technologies Used

- Python
- Flask
- Hugging Face Transformers (`DialoGPT`)
- PyTorch

### 🚀 How to Run

1. Install dependencies:
    ```bash
    pip install flask transformers torch
    ```

2. Run the Flask app:
    ```bash
    python chatbot.py
    ```

3. Send a POST request to:
    ```
    POST http://127.0.0.1:5000/chat
    ```

    Example Request:
    ```json
    {
      "message": "Hello, I need help with my order."
    }
    ```

---

## 📧 Project 2: Spam Email Detection

### 🧾 Description

This project builds a spam detection system using the SMS Spam Collection Dataset. It preprocesses the data and applies a Naive Bayes classifier to categorize messages as spam or not spam.

### 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- Naive Bayes

### 🚀 How to Run

1. Install dependencies:
    ```bash
    pip install pandas scikit-learn
    ```

2. Ensure the `spam.csv` dataset is in the root directory.

3. Run the spam detection script:
    ```bash
    python spam_detector.py
    ```

---

