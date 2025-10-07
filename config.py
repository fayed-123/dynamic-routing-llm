# config.py

# --- Model Configurations ---
MODEL_CONFIGS = {
    'simple': {
        'name': 'Simple Model',
        'max_tokens': 150,
        'response_time': 0.5,
        'resource_cost': 1,
        'accuracy_score': 0.7,
        'description': 'Fast and lightweight model for simple queries'
    },
    'medium': {
        'name': 'Medium Model',
        'max_tokens': 500,
        'response_time': 2.0,
        'resource_cost': 3,
        'accuracy_score': 0.85,
        'description': 'Balanced model for moderately complex queries'
    },
    'advanced': {
        'name': 'Advanced Model',
        'max_tokens': 2000,
        'response_time': 8.0,
        'resource_cost': 10,
        'accuracy_score': 0.95,
        'description': 'High-capability model for complex analysis and reasoning'
    }
}

# --- Query Classification Criteria ---
CLASSIFICATION_CRITERIA = {
    'query_length_thresholds': {
        'simple': 50,
        'medium': 200,
        'advanced': 1000
    },
    'complexity_keywords': {
        'simple': ['what', 'who', 'when', 'where', 'define', 'meaning'],
        'medium': ['how', 'why', 'explain', 'compare', 'describe', 'steps'],
        'advanced': ['analyze', 'evaluate', 'design', 'algorithm', 'optimize']
    },
    'technical_domains': {
        'programming': ['code', 'python', 'java', 'algorithm', 'debug'],
        'mathematics': ['equation', 'calculate', 'math', 'formula'],
        'science': ['research', 'analysis', 'experiment', 'hypothesis']
    }
}

# --- Caching System Settings ---
CACHE_SETTINGS = {
    'enabled': True,
    'max_size': 1000,
    'ttl_seconds': 3600,
    'similarity_threshold': 0.8
}

# --- Fallback Strategy Settings ---
FALLBACK_SETTINGS = {
    'max_retries': 2,
    'retry_delay': 1.0,
    'escalation_strategy': 'step_up',
    'timeout_thresholds': {
        'simple': 2.0,
        'medium': 5.0,
        'advanced': 15.0
    }
}

# --- General System Settings ---
GENERAL_SETTINGS = {
    'default_model': 'medium',
    'verbose_logging': True,
    'debug_mode': False
}

# --- File Paths Configuration ---
FILE_PATHS = {
    'cache_file': 'cache_data.json',
    'logs_file': 'routing_logs.txt',
    'test_queries_file': 'test_queries.json'
}

# --- Helper functions for configuration access ---
def get_model_config(model_name):
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['medium'])

def is_debug_mode():
    return GENERAL_SETTINGS.get('debug_mode', False)

def get_cache_config():
    return CACHE_SETTINGS

def get_fallback_config():
    return FALLBACK_SETTINGS

def get_classification_criteria():
    return CLASSIFICATION_CRITERIA

def get_all_model_names():
    return list(MODEL_CONFIGS.keys())

def get_default_model():
    return GENERAL_SETTINGS.get('default_model', 'medium')