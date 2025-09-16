"""
Dynamic Routing System Configuration
Settings and parameters for the LLM routing system
"""

# Model configurations for different capability levels
MODEL_CONFIGS = {
    'simple': {
        'name': 'Simple Model',
        'max_tokens': 150,
        'response_time': 0.5,  # seconds
        'resource_cost': 1,    # cost points
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

# Query classification criteria
CLASSIFICATION_CRITERIA = {
    'query_length_thresholds': {
        'simple': 50,      # less than 50 characters
        'medium': 200,     # 50-200 characters
        'advanced': 1000   # more than 200 characters
    },

    'complexity_keywords': {
        'simple': [
            'what', 'who', 'when', 'where', 'yes', 'no', 'is', 'are',
            'define', 'meaning', 'simple', 'basic'
        ],
        'medium': [
            'how', 'why', 'explain', 'compare', 'difference', 'between',
            'describe', 'steps', 'process', 'method'
        ],
        'advanced': [
            'analyze', 'evaluate', 'design', 'algorithm', 'strategy',
            'research', 'complex', 'detailed', 'comprehensive', 'optimize'
        ]
    },

    'technical_domains': {
        'programming': ['code', 'python', 'java', 'algorithm', 'debug', 'function'],
        'mathematics': ['equation', 'calculate', 'math', 'formula', 'solve'],
        'science': ['research', 'analysis', 'study', 'experiment', 'hypothesis'],
        'creative': ['write', 'story', 'poem', 'creative', 'generate', 'compose']
    }
}

# Caching system settings
CACHE_SETTINGS = {
    'enabled': True,
    'max_size': 1000,           # maximum number of cached queries
    'ttl_seconds': 3600,        # time to live in seconds (1 hour)
    'similarity_threshold': 0.8  # similarity threshold for similar queries
}

# Fallback strategy settings
FALLBACK_SETTINGS = {
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
    'escalation_strategy': 'step_up',  # 'step_up' or 'direct_to_advanced'
    'timeout_thresholds': {
        'simple': 2.0,
        'medium': 5.0,
        'advanced': 15.0
    }
}

# Evaluation and monitoring settings
EVALUATION_SETTINGS = {
    'log_all_queries': True,
    'performance_metrics': ['response_time', 'accuracy', 'resource_cost'],
    'save_results_to_file': True,
    'results_file_path': 'evaluation_results.json'
}

# General system settings
GENERAL_SETTINGS = {
    'default_model': 'medium',
    'verbose_logging': True,
    'debug_mode': False,
    'language': 'en'
}

# File paths configuration
FILE_PATHS = {
    'cache_file': 'cache_data.json',
    'logs_file': 'routing_logs.txt',
    'test_queries_file': 'test_queries.json',
    'results_file': 'evaluation_results.json'
}

# System messages and notifications
MESSAGES = {
    'model_selected': 'Model selected: {}',
    'fallback_triggered': 'Fallback strategy triggered',
    'cache_hit': 'Cache hit found',
    'cache_miss': 'Cache miss - processing new query',
    'processing_query': 'Processing query...',
    'error_occurred': 'Error occurred: {}',
    'timeout_exceeded': 'Timeout exceeded for model: {}',
    'escalating_model': 'Escalating to higher model: {}',
    'query_completed': 'Query completed successfully'
}

# Helper functions for configuration access
def get_model_config(model_name):
    """Get configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['medium'])

def get_message(key):
    """Get system message by key"""
    return MESSAGES.get(key, key)

def is_debug_mode():
    """Check if debug mode is enabled"""
    return GENERAL_SETTINGS.get('debug_mode', False)

def get_cache_config():
    """Get caching configuration"""
    return CACHE_SETTINGS

def get_fallback_config():
    """Get fallback strategy configuration"""
    return FALLBACK_SETTINGS

def get_classification_criteria():
    """Get query classification criteria"""
    return CLASSIFICATION_CRITERIA

def get_all_model_names():
    """Get list of all available model names"""
    return list(MODEL_CONFIGS.keys())

def get_default_model():
    """Get the default model name"""
    return GENERAL_SETTINGS.get('default_model', 'medium')