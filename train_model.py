# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from query_classifier import QueryClassifier
from test_queries import TestQueryManager

def train():
    """Main function to train and save the classification model."""
    print("🤖 Starting Model Training Process...")

    # 1. Collect and prepare data
    print("Step 1: Collecting and preparing data...")
    query_manager = TestQueryManager()
    temp_classifier = QueryClassifier() # Used only for its feature extraction
    all_queries = query_manager.get_query_collection('all')

    features_list = []
    labels = []
    for item in all_queries:
        if not item.get('query'): continue # Skip empty queries

        features = temp_classifier._extract_features(item['query'])
        feature_row = [
            features['word_count'],
            features['complexity_keywords']['simple'],
            features['complexity_keywords']['medium'],
            features['complexity_keywords']['advanced']
        ]
        features_list.append(feature_row)
        labels.append(item['expected_complexity'])

    feature_names = ['word_count', 'simple_keywords', 'medium_keywords', 'advanced_keywords']
    df = pd.DataFrame(features_list, columns=feature_names)
    df['label'] = labels
    print(f"Data prepared successfully with {len(df)} samples.")

    # 2. Train the model
    print("Step 2: Training the machine learning model...")
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Model training complete. Accuracy on test data: {accuracy:.2%}")

    # 3. Save the model
    print("Step 3: Saving the trained model...")
    joblib.dump(model, 'classifier_model.pkl')
    joblib.dump(list(X.columns), 'model_features.pkl')
    print("💾 Model and features saved successfully.")

if __name__ == '__main__':
    train()