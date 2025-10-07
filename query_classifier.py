# query_classifier.py

import re
import joblib
import pandas as pd
from config import get_classification_criteria, get_all_model_names, get_default_model

class QueryClassifier:
    """Classifies queries using a pre-trained ML model or a rule-based fallback."""
    def __init__(self):
        self.criteria = get_classification_criteria()
        self.default_model = get_default_model()
        self.stats = {
            'total_queries_classified': 0,
            'model_recommendations': {model: 0 for model in get_all_model_names()}
        }

        try:
            self.model = joblib.load('classifier_model.pkl')
            self.feature_names = joblib.load('model_features.pkl')
            self.mode = 'ml'
            print("🤖 ML-based QueryClassifier initialized.")
        except FileNotFoundError:
            self.model = None
            self.feature_names = None
            self.mode = 'rule-based'
            print("⚠️ ML model not found. Falling back to Rule-Based QueryClassifier.")

    def classify_query(self, query: str):
        """Main classification method."""
        if not query or not query.strip():
            return {'recommended_model': self.default_model, 'reasoning': 'Empty query'}

        self.stats['total_queries_classified'] += 1

        if self.mode == 'ml':
            return self._classify_with_ml(query)
        else:
            return self._classify_with_rules(query)

    def _classify_with_ml(self, query: str):
        """Classify using the pre-trained machine learning model."""
        features = self._extract_features(query)
        feature_row = {
            'word_count': features['word_count'],
            'simple_keywords': features['complexity_keywords']['simple'],
            'medium_keywords': features['complexity_keywords']['medium'],
            'advanced_keywords': features['complexity_keywords']['advanced']
        }
        live_data = pd.DataFrame([feature_row], columns=self.feature_names)

        prediction = self.model.predict(live_data)
        recommended_model = prediction[0]

        probabilities = self.model.predict_proba(live_data)
        confidence = probabilities.max()

        self.stats['model_recommendations'][recommended_model] += 1

        return {
            'recommended_model': recommended_model,
            'confidence': confidence,
            'reasoning': f"ML Prediction (Confidence: {confidence:.2%})",
        }

    def _classify_with_rules(self, query: str):
        """Fallback classification using a rule-based heuristic system."""
        features = self._extract_features(query)
        score = self._calculate_complexity_score(features)
        recommended_model = self._determine_model_from_score(score)

        self.stats['model_recommendations'][recommended_model] += 1

        return {
            'recommended_model': recommended_model,
            'confidence': 0.5, # Default confidence for rules
            'reasoning': f"Rule-Based (Score: {score:.2f})",
        }

    def _extract_features(self, query: str) -> dict:
        """Extracts features from the query text."""
        query_lower = query.lower().strip()
        return {
            'word_count': len(query.split()),
            'complexity_keywords': self._count_keywords(query_lower),
        }

    def _count_keywords(self, query_lower: str) -> dict:
        """Counts complexity keywords in the query."""
        counts = {'simple': 0, 'medium': 0, 'advanced': 0}
        for level, keywords in self.criteria['complexity_keywords'].items():
            for keyword in keywords:
                counts[level] += len(re.findall(r'\b' + re.escape(keyword) + r'\b', query_lower))
        return counts

    def _calculate_complexity_score(self, features: dict) -> float:
        """Calculates a heuristic complexity score."""
        score = 0.0
        score += min(features['word_count'] * 0.01, 0.2)
        score += features['complexity_keywords']['simple'] * 0.05
        score += features['complexity_keywords']['medium'] * 0.15
        score += features['complexity_keywords']['advanced'] * 0.30
        return min(score, 1.0)

    def _determine_model_from_score(self, score: float) -> str:
        """Determines model based on complexity score."""
        if score >= 0.70:
            return 'advanced'
        elif score >= 0.35:
            return 'medium'
        else:
            return 'simple'

    def get_classification_stats(self):
        """Returns the statistics for the classifier."""
        return self.stats.copy()