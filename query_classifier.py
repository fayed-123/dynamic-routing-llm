"""
Query Classification System
Analyzes queries and determines the most appropriate model to use
"""

import re
import string
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except ImportError:
    def flesch_reading_ease(text): return 50.0
    def flesch_kincaid_grade(text): return 8.0
from config import get_classification_criteria, get_all_model_names, get_default_model
class QueryClassifier:
    """Classifies queries and recommends appropriate models"""

    def __init__(self):
        self.criteria = get_classification_criteria()
        self.available_models = get_all_model_names()
        self.default_model = get_default_model()

        # Statistics tracking
        self.classification_stats = {
            'total_queries': 0,
            'model_recommendations': {model: 0 for model in self.available_models},
            'classification_reasons': {}
        }

    def classify_query(self, query):
        """Main classification method that returns recommended model and reasoning"""
        if not query or not query.strip():
            return {
                'recommended_model': self.default_model,
                'confidence': 0.0,
                'reasoning': 'Empty query - using default model',
                'complexity_score': 0.0,
                'features': {}
            }

        # Extract query features
        features = self._extract_features(query)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(features)

        # Determine recommended model
        recommended_model = self._determine_model(complexity_score, features)

        # Calculate confidence
        confidence = self._calculate_confidence(complexity_score, features)

        # Generate reasoning
        reasoning = self._generate_reasoning(features, complexity_score, recommended_model)

        # Update statistics
        self._update_stats(recommended_model, reasoning)

        return {
            'recommended_model': recommended_model,
            'confidence': confidence,
            'reasoning': reasoning,
            'complexity_score': complexity_score,
            'features': features
        }

    def _extract_features(self, query):
        """Extract various features from the query"""
        query_clean = query.strip()
        query_lower = query_clean.lower()

        features = {
            # Basic features
            'length': len(query_clean),
            'word_count': len(query_clean.split()),
            'sentence_count': len([s for s in query_clean.split('.') if s.strip()]),

            # Character analysis
            'punctuation_count': sum(1 for char in query_clean if char in string.punctuation),
            'uppercase_ratio': sum(1 for char in query_clean if char.isupper()) / max(len(query_clean), 1),
            'digit_count': sum(1 for char in query_clean if char.isdigit()),

            # Readability scores
            'readability_score': self._get_readability_score(query_clean),
            'grade_level': self._get_grade_level(query_clean),

            # Keyword analysis
            'complexity_keywords': self._count_complexity_keywords(query_lower),
            'domain_keywords': self._identify_domain_keywords(query_lower),

            # Question type
            'question_type': self._identify_question_type(query_lower),
            'has_question_mark': '?' in query_clean,

            # Technical indicators
            'has_code_indicators': self._has_code_indicators(query_clean),
            'has_math_indicators': self._has_math_indicators(query_clean),
            'has_technical_terms': self._has_technical_terms(query_lower)
        }

        return features

    def _get_readability_score(self, text):
        """Calculate readability score using Flesch Reading Ease"""
        try:
            return flesch_reading_ease(text)
        except:
            return 50.0  # Default moderate score

    def _get_grade_level(self, text):
        """Calculate grade level using Flesch-Kincaid"""
        try:
            return flesch_kincaid_grade(text)
        except:
            return 8.0  # Default grade level

    def _count_complexity_keywords(self, query_lower):
        """Count keywords by complexity level"""
        counts = {'simple': 0, 'medium': 0, 'advanced': 0}

        for level, keywords in self.criteria['complexity_keywords'].items():
            for keyword in keywords:
                counts[level] += query_lower.count(keyword.lower())

        return counts

    def _identify_domain_keywords(self, query_lower):
        """Identify technical domain keywords"""
        domains = {}

        for domain, keywords in self.criteria['technical_domains'].items():
            count = sum(query_lower.count(keyword.lower()) for keyword in keywords)
            if count > 0:
                domains[domain] = count

        return domains

    def _identify_question_type(self, query_lower):
        """Identify the type of question being asked"""
        question_patterns = {
            'factual': ['what', 'who', 'when', 'where', 'which'],
            'procedural': ['how', 'steps', 'process', 'method', 'way'],
            'analytical': ['why', 'analyze', 'compare', 'evaluate', 'assess'],
            'creative': ['create', 'design', 'generate', 'write', 'compose'],
            'computational': ['calculate', 'solve', 'compute', 'algorithm']
        }

        for q_type, keywords in question_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return q_type

        return 'general'

    def _has_code_indicators(self, query):
        """Check if query contains code-related indicators"""
        code_patterns = [
            r'\b(def|class|import|from|if|else|for|while)\b',
            r'[{}()\[\]<>]',
            r'\b(python|java|javascript|html|css|sql)\b',
            r'[;{}]',
            r'\b(function|variable|array|object)\b'
        ]

        return any(re.search(pattern, query.lower()) for pattern in code_patterns)

    def _has_math_indicators(self, query):
        """Check if query contains mathematical indicators"""
        math_patterns = [
            r'[+\-*/=<>]',
            r'\b(equation|formula|calculate|solve|derivative|integral)\b',
            r'\b(sum|product|average|mean|median)\b',
            r'[∫∑∏√±≤≥≠]',
            r'\b(x|y|z)\s*[=<>]'
        ]

        return any(re.search(pattern, query.lower()) for pattern in math_patterns)

    def _has_technical_terms(self, query_lower):
        """Check for general technical terminology"""
        technical_terms = [
            'algorithm', 'database', 'network', 'server', 'client',
            'protocol', 'framework', 'library', 'api', 'interface',
            'architecture', 'optimization', 'performance', 'scalability',
            'security', 'encryption', 'authentication', 'deployment'
        ]

        return any(term in query_lower for term in technical_terms)

    def _calculate_complexity_score(self, features):
        """Calculate overall complexity score (0.0 to 1.0)"""
        score = 0.0

        # Length-based scoring (0.0-0.3)
        length_thresholds = self.criteria['query_length_thresholds']
        if features['length'] < length_thresholds['simple']:
            score += 0.1
        elif features['length'] < length_thresholds['medium']:
            score += 0.2
        else:
            score += 0.3

        # Keyword-based scoring (0.0-0.3)
        keyword_counts = features['complexity_keywords']
        if keyword_counts['advanced'] > 0:
            score += 0.3
        elif keyword_counts['medium'] > 0:
            score += 0.2
        elif keyword_counts['simple'] > 0:
            score += 0.1
        else:
            score += 0.15  # Default for no specific keywords

        # Question type scoring (0.0-0.2)
        question_type_scores = {
            'factual': 0.05,
            'general': 0.1,
            'procedural': 0.15,
            'analytical': 0.2,
            'creative': 0.18,
            'computational': 0.2
        }
        score += question_type_scores.get(features['question_type'], 0.1)

        # Technical indicators (0.0-0.2)
        if features['has_code_indicators']:
            score += 0.1
        if features['has_math_indicators']:
            score += 0.1
        if features['has_technical_terms']:
            score += 0.05
        if features['domain_keywords']:
            score += 0.05

        return min(score, 1.0)

    def _determine_model(self, complexity_score, features):
        """Determine the recommended model based on complexity score and features"""
        # Force advanced model for specific indicators
        if (features['has_code_indicators'] and features['word_count'] > 20) or \
                features['complexity_keywords']['advanced'] > 2 or \
                features['question_type'] in ['analytical', 'computational'] and features['word_count'] > 15:
            return 'advanced'

        # Force simple model for basic queries
        if complexity_score < 0.3 and features['word_count'] < 10 and \
                features['question_type'] == 'factual':
            return 'simple'

        # Score-based determination
        if complexity_score >= 0.7:
            return 'advanced'
        elif complexity_score >= 0.4:
            return 'medium'
        else:
            return 'simple'

    def _calculate_confidence(self, complexity_score, features):
        """Calculate confidence in the classification (0.0 to 1.0)"""
        confidence = 0.5  # Base confidence

        # Strong indicators increase confidence
        if features['complexity_keywords']['advanced'] > 0:
            confidence += 0.3
        elif features['complexity_keywords']['simple'] > 0:
            confidence += 0.2

        # Clear question types increase confidence
        if features['question_type'] != 'general':
            confidence += 0.2

        # Technical indicators
        if features['has_code_indicators'] or features['has_math_indicators']:
            confidence += 0.15

        # Length consistency
        length = features['length']
        thresholds = self.criteria['query_length_thresholds']
        if (complexity_score < 0.4 and length < thresholds['simple']) or \
                (complexity_score > 0.7 and length > thresholds['medium']):
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_reasoning(self, features, complexity_score, recommended_model):
        """Generate human-readable reasoning for the classification"""
        reasons = []

        # Length reasoning
        length = features['length']
        if length < 50:
            reasons.append(f"Short query ({length} chars)")
        elif length > 200:
            reasons.append(f"Long query ({length} chars)")

        # Keyword reasoning
        keyword_counts = features['complexity_keywords']
        if keyword_counts['advanced'] > 0:
            reasons.append(f"Contains {keyword_counts['advanced']} advanced keywords")
        elif keyword_counts['medium'] > 0:
            reasons.append(f"Contains {keyword_counts['medium']} medium complexity keywords")

        # Question type reasoning
        if features['question_type'] != 'general':
            reasons.append(f"Question type: {features['question_type']}")

        # Technical reasoning
        if features['has_code_indicators']:
            reasons.append("Contains code-related content")
        if features['has_math_indicators']:
            reasons.append("Contains mathematical content")
        if features['domain_keywords']:
            domains = list(features['domain_keywords'].keys())
            reasons.append(f"Technical domains: {', '.join(domains)}")

        # Complexity score
        reasons.append(f"Complexity score: {complexity_score:.2f}")

        return f"Recommended {recommended_model} model. Reasons: {'; '.join(reasons)}"

    def _update_stats(self, recommended_model, reasoning):
        """Update classification statistics"""
        self.classification_stats['total_queries'] += 1
        self.classification_stats['model_recommendations'][recommended_model] += 1

        # Track reasoning patterns
        key_reason = reasoning.split('.')[0]  # First part of reasoning
        if key_reason not in self.classification_stats['classification_reasons']:
            self.classification_stats['classification_reasons'][key_reason] = 0
        self.classification_stats['classification_reasons'][key_reason] += 1

    def get_classification_stats(self):
        """Get classification statistics"""
        return self.classification_stats.copy()

    def reset_stats(self):
        """Reset classification statistics"""
        self.classification_stats = {
            'total_queries': 0,
            'model_recommendations': {model: 0 for model in self.available_models},
            'classification_reasons': {}
        }


# Utility functions
def classify_single_query(query):
    """Convenience function to classify a single query"""
    classifier = QueryClassifier()
    return classifier.classify_query(query)


def batch_classify_queries(queries):
    """Classify multiple queries at once"""
    classifier = QueryClassifier()
    results = []

    for query in queries:
        result = classifier.classify_query(query)
        results.append({
            'query': query,
            'classification': result
        })

    return {
        'results': results,
        'stats': classifier.get_classification_stats()
    }