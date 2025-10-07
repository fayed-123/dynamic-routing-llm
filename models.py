# models.py

import time
import random
from abc import ABC, abstractmethod
from config import get_model_config

class BaseModel(ABC):
    """Abstract base class for all models."""
    def __init__(self, model_name):
        self.name = model_name
        self.config = get_model_config(model_name)
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_response_time': 0.0,
            'total_resource_cost': 0
        }

    @abstractmethod
    def process_query(self, query):
        """Process a query and return a response dictionary."""
        pass

    def get_stats(self):
        """Get model performance statistics."""
        total = max(self.stats['total_queries'], 1)
        return {
            'total_queries': self.stats['total_queries'],
            'success_rate': self.stats['successful_queries'] / total,
            'average_response_time': self.stats['total_response_time'] / total
        }

    def _update_stats(self, success, response_time, resource_cost):
        """Update model statistics after processing a query."""
        self.stats['total_queries'] += 1
        self.stats['total_response_time'] += response_time
        self.stats['total_resource_cost'] += resource_cost
        if success:
            self.stats['successful_queries'] += 1
        else:
            self.stats['failed_queries'] += 1

class SimpleModel(BaseModel):
    """Simple, fast model for basic queries."""
    def __init__(self):
        super().__init__('simple')

    def process_query(self, query):
        start_time = time.time()
        try:
            processing_time = self.config['response_time'] + random.uniform(-0.2, 0.2)
            time.sleep(processing_time)
            response = f"Simple response for: '{query[:30]}...'"
            success = True
        except Exception:
            response = "Error in SimpleModel."
            success = False

        response_time = time.time() - start_time
        resource_cost = self.config['resource_cost'] if success else 0
        self._update_stats(success, response_time, resource_cost)

        return {
            'response': response,
            'response_time': response_time,
            'resource_cost': resource_cost,
            'success': success
        }

class MediumModel(BaseModel):
    """Medium complexity model for balanced performance."""
    def __init__(self):
        super().__init__('medium')

    def process_query(self, query):
        start_time = time.time()
        try:
            processing_time = self.config['response_time'] + random.uniform(-0.5, 0.5)
            time.sleep(processing_time)
            response = f"Medium explanation for: '{query[:40]}...'"
            success = True
        except Exception:
            response = "Error in MediumModel."
            success = False

        response_time = time.time() - start_time
        resource_cost = self.config['resource_cost'] if success else 0
        self._update_stats(success, response_time, resource_cost)

        return {
            'response': response,
            'response_time': response_time,
            'resource_cost': resource_cost,
            'success': success
        }

class AdvancedModel(BaseModel):
    """Advanced model for complex analysis and reasoning."""
    def __init__(self):
        super().__init__('advanced')

    def process_query(self, query):
        start_time = time.time()
        try:
            processing_time = self.config['response_time'] + random.uniform(-2.0, 2.0)
            time.sleep(processing_time)
            response = f"Advanced analysis for: '{query[:50]}...'"
            success = True
        except Exception:
            response = "Error in AdvancedModel."
            success = False

        response_time = time.time() - start_time
        resource_cost = self.config['resource_cost'] if success else 0
        self._update_stats(success, response_time, resource_cost)

        return {
            'response': response,
            'response_time': response_time,
            'resource_cost': resource_cost,
            'success': success
        }

class ModelFactory:
    """Factory class for creating model instances."""
    _models = {
        'simple': SimpleModel,
        'medium': MediumModel,
        'advanced': AdvancedModel
    }

    @classmethod
    def create_all_models(cls):
        """Create instances of all available models."""
        return {name: model_class() for name, model_class in cls._models.items()}