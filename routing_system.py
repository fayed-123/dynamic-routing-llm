"""
Dynamic Routing System
Main system that orchestrates query classification, model selection,
caching, and fallback strategies
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from query_classifier import QueryClassifier
from models import ModelFactory, BaseModel
from cache_manager import CacheManager
from config import (
    get_fallback_config, get_message, is_debug_mode,
    get_default_model, GENERAL_SETTINGS, FILE_PATHS
)


class RoutingDecision:
    """Represents a routing decision with all metadata"""

    def __init__(self):
        self.timestamp = time.time()
        self.query = ""
        self.original_query = ""
        self.classification_result = None
        self.recommended_model = ""
        self.actual_model_used = ""
        self.fallback_triggered = False
        self.fallback_reason = ""
        self.cache_hit = False
        self.cache_similarity_match = False
        self.processing_time = 0.0
        self.total_time = 0.0
        self.success = False
        self.error_message = ""
        self.resource_cost = 0
        self.response = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert routing decision to dictionary"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'query': self.query,
            'classification': {
                'recommended_model': self.recommended_model,
                'confidence': self.classification_result.get('confidence', 0) if self.classification_result else 0,
                'complexity_score': self.classification_result.get('complexity_score', 0) if self.classification_result else 0,
                'reasoning': self.classification_result.get('reasoning', '') if self.classification_result else ''
            },
            'execution': {
                'actual_model_used': self.actual_model_used,
                'cache_hit': self.cache_hit,
                'cache_similarity_match': self.cache_similarity_match,
                'fallback_triggered': self.fallback_triggered,
                'fallback_reason': self.fallback_reason,
                'processing_time': self.processing_time,
                'total_time': self.total_time,
                'resource_cost': self.resource_cost
            },
            'result': {
                'success': self.success,
                'error_message': self.error_message,
                'response_preview': self.response[:100] + '...' if len(self.response) > 100 else self.response
            }
        }


class DynamicRoutingSystem:
    """Main dynamic routing system that coordinates all components"""

    def __init__(self):
        # Initialize components
        self.classifier = QueryClassifier()
        self.cache_manager = CacheManager()
        self.models = ModelFactory.create_all_models()
        self.fallback_config = get_fallback_config()

        # System state
        self.is_running = True
        self.routing_history: List[RoutingDecision] = []

        # Statistics
        self.system_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'fallback_triggers': 0,
            'model_usage': {model: 0 for model in self.models.keys()},
            'total_processing_time': 0.0,
            'total_resource_cost': 0,
            'start_time': time.time()
        }

        # Setup logging
        self._setup_logging()
        self.logger.info("Dynamic Routing System initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.DEBUG if is_debug_mode() else logging.INFO

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(FILE_PATHS['logs_file']),
                logging.StreamHandler() if GENERAL_SETTINGS['verbose_logging'] else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger('DynamicRoutingSystem')

    def process_query(self, query: str, force_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to process a query through the routing system

        Args:
            query: The input query to process
            force_model: Optional model name to force use (bypasses classification)

        Returns:
            Dictionary containing the result and all routing metadata
        """
        start_time = time.time()
        decision = RoutingDecision()
        decision.query = query.strip()
        decision.original_query = query

        try:
            self.logger.info(f"Processing query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

            # Step 1: Check cache first
            cached_result = self._check_cache(decision)
            if cached_result:
                return cached_result

            # Step 2: Classify query (unless model is forced)
            if force_model:
                decision.recommended_model = force_model
                decision.classification_result = {'confidence': 1.0, 'reasoning': 'Force model specified'}
                self.logger.info(f"Force using model: {force_model}")
            else:
                self._classify_query(decision)

            # Step 3: Route to appropriate model with fallback
            result = self._route_with_fallback(decision)

            # Step 4: Cache successful results
            if result['success']:
                self._cache_result(decision, result)

            # Step 5: Update statistics and history
            decision.total_time = time.time() - start_time
            self._update_stats(decision)
            self.routing_history.append(decision)

            self.logger.info(f"Query processed successfully in {decision.total_time:.3f}s using {decision.actual_model_used}")

            return self._format_final_result(decision, result)

        except Exception as e:
            decision.success = False
            decision.error_message = str(e)
            decision.total_time = time.time() - start_time

            self.logger.error(f"Error processing query: {e}")
            self._update_stats(decision)
            self.routing_history.append(decision)

            return self._format_error_result(decision, str(e))

    def _check_cache(self, decision: RoutingDecision) -> Optional[Dict[str, Any]]:
        """Check cache for existing results"""
        cached_result = self.cache_manager.get(decision.query)

        if cached_result:
            decision.cache_hit = True
            decision.cache_similarity_match = cached_result.get('similarity_match', False)
            decision.actual_model_used = cached_result['model_used']
            decision.success = True
            decision.response = cached_result['result'].get('response', '')
            decision.resource_cost = 0  # Cache hits have no resource cost

            self.logger.info(f"Cache hit found (similarity: {cached_result.get('similarity_match', False)})")

            return self._format_cached_result(decision, cached_result)

        return None

    def _classify_query(self, decision: RoutingDecision):
        """Classify the query to determine recommended model"""
        classification_start = time.time()

        decision.classification_result = self.classifier.classify_query(decision.query)
        decision.recommended_model = decision.classification_result['recommended_model']

        classification_time = time.time() - classification_start

        self.logger.debug(f"Classification completed in {classification_time:.3f}s: "
                          f"model={decision.recommended_model}, "
                          f"confidence={decision.classification_result['confidence']:.2f}")

    def _route_with_fallback(self, decision: RoutingDecision) -> Dict[str, Any]:
        """Route query to model with fallback strategy"""
        current_model = decision.recommended_model
        retry_count = 0
        max_retries = self.fallback_config['max_retries']

        while retry_count <= max_retries:
            try:
                # Get the model instance
                if current_model not in self.models:
                    raise ValueError(f"Model '{current_model}' not available")

                model = self.models[current_model]

                # Process with timeout
                result = self._process_with_timeout(model, decision.query, current_model)

                # Success - record decision and return result
                decision.actual_model_used = current_model
                decision.success = True
                decision.processing_time = result['response_time']
                decision.resource_cost = result['resource_cost']
                decision.response = result['response']

                return result

            except TimeoutError:
                self.logger.warning(f"Timeout with model {current_model}, attempt {retry_count + 1}")
                decision.fallback_reason = f"Timeout with {current_model}"

            except Exception as e:
                self.logger.warning(f"Error with model {current_model}: {e}, attempt {retry_count + 1}")
                decision.fallback_reason = f"Error with {current_model}: {str(e)}"

            # Fallback logic
            retry_count += 1
            if retry_count <= max_retries:
                decision.fallback_triggered = True
                current_model = self._get_fallback_model(current_model, retry_count)

                self.logger.info(f"Fallback triggered: switching to {current_model}")

                # Wait before retry
                time.sleep(self.fallback_config['retry_delay'])

        # All retries failed
        decision.success = False
        decision.error_message = f"All models failed after {max_retries} retries"
        raise Exception(decision.error_message)

    def _process_with_timeout(self, model: BaseModel, query: str, model_name: str) -> Dict[str, Any]:
        """Process query with timeout protection"""
        timeout = self.fallback_config['timeout_thresholds'].get(model_name, 30.0)

        # Simple timeout implementation (for demonstration)
        # In production, you might want to use threading or async/await
        start_time = time.time()
        result = model.process_query(query)
        processing_time = time.time() - start_time

        if processing_time > timeout:
            raise TimeoutError(f"Model {model_name} exceeded timeout of {timeout}s")

        return result

    def _get_fallback_model(self, current_model: str, retry_count: int) -> str:
        """Determine fallback model based on strategy"""
        strategy = self.fallback_config['escalation_strategy']
        available_models = list(self.models.keys())

        if strategy == 'step_up':
            # Try progressively more powerful models
            model_hierarchy = ['simple', 'medium', 'advanced']
            current_index = model_hierarchy.index(current_model) if current_model in model_hierarchy else 0
            next_index = min(current_index + 1, len(model_hierarchy) - 1)
            return model_hierarchy[next_index]

        elif strategy == 'direct_to_advanced':
            # Go directly to the most powerful model
            return 'advanced'

        else:
            # Default: try next available model
            current_index = available_models.index(current_model) if current_model in available_models else 0
            next_index = (current_index + 1) % len(available_models)
            return available_models[next_index]

    def _cache_result(self, decision: RoutingDecision, result: Dict[str, Any]):
        """Cache the successful result"""
        try:
            success = self.cache_manager.put(
                decision.query,
                result,
                decision.actual_model_used
            )

            if success:
                self.logger.debug("Result cached successfully")
            else:
                self.logger.warning("Failed to cache result")

        except Exception as e:
            self.logger.error(f"Error caching result: {e}")

    def _update_stats(self, decision: RoutingDecision):
        """Update system statistics"""
        self.system_stats['total_queries'] += 1

        if decision.success:
            self.system_stats['successful_queries'] += 1
        else:
            self.system_stats['failed_queries'] += 1

        if decision.cache_hit:
            self.system_stats['cache_hits'] += 1

        if decision.fallback_triggered:
            self.system_stats['fallback_triggers'] += 1

        if decision.actual_model_used:
            self.system_stats['model_usage'][decision.actual_model_used] += 1

        self.system_stats['total_processing_time'] += decision.processing_time
        self.system_stats['total_resource_cost'] += decision.resource_cost

    def _format_final_result(self, decision: RoutingDecision, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final result for return"""
        return {
            'success': True,
            'response': result['response'],
            'routing_info': {
                'query': decision.query,
                'recommended_model': decision.recommended_model,
                'actual_model_used': decision.actual_model_used,
                'cache_hit': decision.cache_hit,
                'fallback_triggered': decision.fallback_triggered,
                'processing_time': decision.processing_time,
                'total_time': decision.total_time,
                'resource_cost': decision.resource_cost,
                'classification': decision.classification_result
            },
            'metadata': {
                'timestamp': decision.timestamp,
                'system_stats': self.get_quick_stats()
            }
        }

    def _format_cached_result(self, decision: RoutingDecision, cached_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format cached result for return"""
        return {
            'success': True,
            'response': cached_result['result']['response'],
            'routing_info': {
                'query': decision.query,
                'actual_model_used': decision.actual_model_used,
                'cache_hit': True,
                'cache_similarity_match': decision.cache_similarity_match,
                'cached_at': cached_result['cached_at'],
                'access_count': cached_result['access_count'],
                'processing_time': 0.0,
                'total_time': decision.total_time,
                'resource_cost': 0
            },
            'metadata': {
                'timestamp': decision.timestamp,
                'cache_info': {
                    'original_query': cached_result.get('original_query', decision.query),
                    'similarity_score': cached_result.get('similarity_score', 1.0)
                }
            }
        }

    def _format_error_result(self, decision: RoutingDecision, error_message: str) -> Dict[str, Any]:
        """Format error result for return"""
        return {
            'success': False,
            'error': error_message,
            'routing_info': {
                'query': decision.query,
                'recommended_model': decision.recommended_model,
                'fallback_triggered': decision.fallback_triggered,
                'fallback_reason': decision.fallback_reason,
                'total_time': decision.total_time
            },
            'metadata': {
                'timestamp': decision.timestamp,
                'system_stats': self.get_quick_stats()
            }
        }

    def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick system statistics"""
        uptime = time.time() - self.system_stats['start_time']
        total_queries = max(self.system_stats['total_queries'], 1)

        return {
            'uptime_seconds': uptime,
            'total_queries': self.system_stats['total_queries'],
            'success_rate': self.system_stats['successful_queries'] / total_queries,
            'cache_hit_rate': self.system_stats['cache_hits'] / total_queries,
            'fallback_rate': self.system_stats['fallback_triggers'] / total_queries
        }

    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed system statistics"""
        return {
            'system': self.system_stats.copy(),
            'cache': self.cache_manager.get_stats(),
            'classification': self.classifier.get_classification_stats(),
            'models': {name: model.get_stats() for name, model in self.models.items()},
            'recent_queries': [decision.to_dict() for decision in self.routing_history[-10:]]
        }

    def reset_stats(self):
        """Reset all system statistics"""
        self.system_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'fallback_triggers': 0,
            'model_usage': {model: 0 for model in self.models.keys()},
            'total_processing_time': 0.0,
            'total_resource_cost': 0,
            'start_time': time.time()
        }
        self.routing_history.clear()
        self.classifier.reset_stats()

    def shutdown(self):
        """Graceful shutdown of the routing system"""
        self.logger.info("Shutting down Dynamic Routing System...")

        # Save cache
        self.cache_manager.save_cache_to_file()

        # Log final statistics
        self.logger.info(f"Final stats: {self.get_quick_stats()}")

        self.is_running = False
        self.logger.info("Dynamic Routing System shutdown complete")


# Utility functions for easy usage
def create_routing_system() -> DynamicRoutingSystem:
    """Factory function to create a routing system instance"""
    return DynamicRoutingSystem()


def quick_query(query: str) -> str:
    """Quick utility to process a single query and return just the response"""
    system = create_routing_system()
    result = system.process_query(query)

    if result['success']:
        return result['response']
    else:
        return f"Error: {result['error']}"


if __name__ == "__main__":
    print("Testing Dynamic Routing System...")

    # Create system instance
    system = create_routing_system()

    # Test queries
    test_queries = [
        "What is Python?",
        "How do I implement a binary search algorithm?",
        "Analyze the computational complexity of different sorting algorithms and design an optimal solution for large datasets"
    ]

    print("\nProcessing test queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")

        result = system.process_query(query)

        if result['success']:
            print(f"Model Used: {result['routing_info']['actual_model_used']}")
            print(f"Cache Hit: {result['routing_info']['cache_hit']}")
            print(f"Processing Time: {result['routing_info']['processing_time']:.3f}s")
            print(f"Response: {result['response'][:100]}...")
        else:
            print(f"Error: {result['error']}")

    # Print system statistics
    print(f"\n--- System Statistics ---")
    stats = system.get_quick_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test cache by running same query again
    print(f"\n--- Testing Cache (repeating first query) ---")
    result = system.process_query(test_queries[0])
    print(f"Cache Hit: {result['routing_info']['cache_hit']}")

    # Shutdown system
    system.shutdown()