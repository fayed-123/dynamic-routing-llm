# routing_system.py

import time
import logging
import sys
from typing import Dict, Any, Optional

from query_classifier import QueryClassifier
from models import ModelFactory, BaseModel
from cache_manager import CacheManager
from config import get_fallback_config, GENERAL_SETTINGS, FILE_PATHS

class DynamicRoutingSystem:
    """The central orchestrator for processing LLM queries."""
    def __init__(self):
        self.classifier = QueryClassifier()
        self.cache_manager = CacheManager()
        self.models = ModelFactory.create_all_models()
        self.fallback_config = get_fallback_config()
        self.stats = {
            'total_queries': 0, 'successful_queries': 0, 'failed_queries': 0,
            'cache_hits': 0, 'fallback_triggers': 0,
            'model_usage': {model: 0 for model in self.models.keys()}
        }
        self._setup_logging()
        self.logger.info("Dynamic Routing System initialized.")

    def _setup_logging(self):
        """Initializes the logger."""
        self.logger = logging.getLogger('DynamicRoutingSystem')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers: # Avoid adding handlers multiple times
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler(FILE_PATHS['logs_file'], encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            if GENERAL_SETTINGS['verbose_logging']:
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(formatter)
                self.logger.addHandler(stream_handler)

    def process_query(self, query: str) -> Dict:
        """
        Processes a query through the full pipeline:
        Cache -> Classifier -> Model -> Fallback -> Cache Save.
        """
        self.stats['total_queries'] += 1
        self.logger.info(f"Processing query: '{query[:70]}...'")

        # 1. Check cache
        cached_result = self.cache_manager.get(query)
        if cached_result:
            self.stats['cache_hits'] += 1
            self.logger.info(f"Cache hit. Model used previously: '{cached_result['model_used']}'.")
            return self._format_result(cached_result, is_cache_hit=True)

        # 2. Classify query
        classification = self.classifier.classify_query(query)
        recommended_model = classification['recommended_model']

        # 3. Route with fallback
        try:
            result, actual_model = self._route_with_fallback(query, recommended_model)
            if result['success']:
                self.stats['successful_queries'] += 1
                self.cache_manager.put(query, result, actual_model)
                self.cache_manager.save_cache_to_file() # Write-through caching
            else:
                self.stats['failed_queries'] += 1

            self.stats['model_usage'][actual_model] += 1
            return self._format_result(result, actual_model=actual_model, classification=classification)

        except Exception as e:
            self.stats['failed_queries'] += 1
            self.logger.error(f"Query failed after all retries: {e}")
            return {'success': False, 'error': str(e)}

    def _route_with_fallback(self, query: str, recommended_model: str) -> (Dict, str):
        """Attempts to process a query, escalating to other models on failure."""
        current_model_name = recommended_model
        for attempt in range(self.fallback_config['max_retries'] + 1):
            try:
                model = self.models[current_model_name]
                timeout = self.fallback_config['timeout_thresholds'][current_model_name]

                # Simplified timeout check for simulation
                start_time = time.time()
                result = model.process_query(query)
                duration = time.time() - start_time
                if duration > timeout:
                    raise TimeoutError(f"Model '{current_model_name}' exceeded timeout of {timeout}s.")

                result['processing_time'] = duration
                return result, current_model_name

            except (Exception, TimeoutError) as e:
                self.logger.warning(f"Attempt {attempt + 1} with '{current_model_name}' failed: {e}")
                if attempt < self.fallback_config['max_retries']:
                    self.stats['fallback_triggers'] += 1
                    current_model_name = self._get_fallback_model(current_model_name)
                    self.logger.info(f"Fallback triggered. Retrying with '{current_model_name}'.")
                    time.sleep(self.fallback_config['retry_delay'])

        raise Exception("Query failed after all fallback attempts.")

    def _get_fallback_model(self, current_model: str) -> str:
        """Determines the next model to use based on the escalation strategy."""
        if self.fallback_config['escalation_strategy'] == 'step_up':
            hierarchy = ['simple', 'medium', 'advanced']
            try:
                idx = hierarchy.index(current_model)
                return hierarchy[min(idx + 1, len(hierarchy) - 1)]
            except ValueError:
                return 'medium' # Default fallback
        return 'advanced' # Default to the most robust model

    def _format_result(self, result: Dict, is_cache_hit=False, actual_model=None, classification=None) -> Dict:
        """Formats the final response dictionary."""
        if is_cache_hit:
            return {
                'success': True,
                'response': result['result']['response'],
                'routing_info': {
                    'actual_model_used': result['model_used'],
                    'cache_hit': True,
                    'fallback_triggered': False,
                    'processing_time': 0.0,
                    'resource_cost': 0,
                    'classification': {'reasoning': 'From cache'}
                }
            }

        return {
            'success': result['success'],
            'response': result.get('response'),
            'error': result.get('error'),
            'routing_info': {
                'actual_model_used': actual_model,
                'cache_hit': False,
                'fallback_triggered': self.stats['fallback_triggers'] > 0,
                'processing_time': result.get('processing_time', 0),
                'resource_cost': result.get('resource_cost', 0),
                'classification': classification
            }
        }

    def get_detailed_stats(self) -> Dict:
        """Returns a comprehensive dictionary of system statistics."""
        return {
            'system': self.stats,
            'cache': self.cache_manager.get_stats(),
            'classification': self.classifier.get_classification_stats()
        }

    def shutdown(self):
        """Performs graceful shutdown procedures."""
        self.logger.info("Shutting down system...")
        self.cache_manager.save_cache_to_file()
        self.logger.info("Shutdown complete.")

def create_routing_system() -> DynamicRoutingSystem:
    """Factory function for the main routing system."""
    return DynamicRoutingSystem()