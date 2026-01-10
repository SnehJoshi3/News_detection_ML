# 🕵️ Fake News Detection System

A Machine Learning project that classifies news articles as either **Real** or **Fake** using Natural Language Processing (NLP) techniques.

## 🚀 Overview
This project processes thousands of news articles, cleans the text data, and trains multiple classification models to detect misinformation with high accuracy.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Regex
* **Techniques:** TF-IDF Vectorization, Text Preprocessing

## 📊 Models Used
* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

## 📂 Dataset
The dataset used consists of two CSV files: `True.csv` and `Fake.csv`, containing news articles labeled as genuine or fake. 
*(Note: If you used the Kaggle dataset, verify the license and link it here).*

## ⚙️ How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/SnehJoshi3/fake-news-detector.git](https://github.com/SnehJoshi3/fake-news-detector.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  Run the script:
    ```bash
    python fake_news_detector.py
    ```
4.  **Interactive Mode:** After training, the script allows you to input any text to test the model in real-time.

## 📈 Results
* **Logistic Regression Accuracy:** ~98%
* **Decision Tree Accuracy:** ~99%
