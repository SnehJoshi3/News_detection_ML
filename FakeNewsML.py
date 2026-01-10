import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Data Cleaning Function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# 2. Main Execution
def main():
    print("Loading datasets...")
    # Ensure these CSV files are in the same folder
    try:
        true_df = pd.read_csv('True.csv')
        fake_df = pd.read_csv('Fake.csv')
    except FileNotFoundError:
        print("Error: 'True.csv' or 'Fake.csv' not found. Please download the dataset.")
        return

    # Labeling: 1 for True, 0 for Fake
    true_df['class'] = 1
    fake_df['class'] = 0

    # Merge and Shuffle
    # Using only last 10 entries for manual testing verification later
    # manual_testing = pd.concat([fake_df.tail(10), true_df.tail(10)], axis=0) 
    
    # Merging main data
    news_df = pd.concat([true_df, fake_df], axis=0)
    news_df = news_df.sample(frac=1).reset_index(drop=True)

    # Drop unnecessary columns
    news_df = news_df.drop(['title', 'subject', 'date'], axis=1)

    print("Preprocessing text...")
    news_df['text'] = news_df['text'].apply(wordopt)

    # Define X and Y
    x = news_df['text']
    y = news_df['class']

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Vectorization
    print("Vectorizing text...")
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # --- Model Training ---
    print("\nTraining Logistic Regression...")
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)
    pred_lr = LR.predict(xv_test)
    print(f"Logistic Regression Accuracy: {LR.score(xv_test, y_test):.4f}")
    print(classification_report(y_test, pred_lr))

    print("\nTraining Decision Tree...")
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    pred_dt = DT.predict(xv_test)
    print(f"Decision Tree Accuracy: {DT.score(xv_test, y_test):.4f}")
    print(classification_report(y_test, pred_dt))
    
    # (You can add Random Forest / Gradient Boosting here similarly)

    # --- Interactive Manual Test ---
    print("\n--- Manual Testing Mode ---")
    def output_label(n):
        return "Fake News" if n == 0 else "True News"

    def manual_testing(news):
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt) 
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)

        return print(f"\nLR Prediction: {output_label(pred_LR[0])} \nDT Prediction: {output_label(pred_DT[0])}")

    while True:
        news = input("\nEnter a news text to verify (or 'exit' to stop): ")
        if news.lower() == 'exit':
            break
        manual_testing(news)

if __name__ == "__main__":
    main()
