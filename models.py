"""
Model definitions for the dynamic routing system
Contains different model implementations with varying capabilities
"""

import time
import random
from abc import ABC, abstractmethod
from config import get_model_config, get_message


class BaseModel(ABC):
    """Abstract base class for all models"""

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
        """Process a query and return response"""
        pass

    def get_model_info(self):
        """Get model configuration information"""
        return {
            'name': self.config['name'],
            'max_tokens': self.config['max_tokens'],
            'expected_response_time': self.config['response_time'],
            'resource_cost': self.config['resource_cost'],
            'accuracy_score': self.config['accuracy_score'],
            'description': self.config['description']
        }

    def get_stats(self):
        """Get model performance statistics"""
        avg_response_time = 0
        if self.stats['total_queries'] > 0:
            avg_response_time = self.stats['total_response_time'] / self.stats['total_queries']

        return {
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'failed_queries': self.stats['failed_queries'],
            'success_rate': self.stats['successful_queries'] / max(self.stats['total_queries'], 1),
            'average_response_time': avg_response_time,
            'total_resource_cost': self.stats['total_resource_cost']
        }

    def _update_stats(self, success, response_time, resource_cost):
        """Update model statistics"""
        self.stats['total_queries'] += 1
        self.stats['total_response_time'] += response_time
        self.stats['total_resource_cost'] += resource_cost

        if success:
            self.stats['successful_queries'] += 1
        else:
            self.stats['failed_queries'] += 1


class SimpleModel(BaseModel):
    """Simple, fast model for basic queries"""

    def __init__(self):
        super().__init__('simple')

    def process_query(self, query):
        """Process query with simple model logic"""
        start_time = time.time()

        try:
            # Simulate processing time
            processing_time = random.uniform(0.3, 0.7)  # 0.3-0.7 seconds
            time.sleep(processing_time)

            # Simple response generation (simulation)
            response = self._generate_simple_response(query)

            # Calculate metrics
            response_time = time.time() - start_time
            resource_cost = self.config['resource_cost']

            # Update statistics
            self._update_stats(True, response_time, resource_cost)

            return {
                'response': response,
                'model_used': self.name,
                'response_time': response_time,
                'resource_cost': resource_cost,
                'accuracy_estimate': self.config['accuracy_score'],
                'success': True
            }

        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time, self.config['resource_cost'])

            return {
                'response': f"Error processing query: {str(e)}",
                'model_used': self.name,
                'response_time': response_time,
                'resource_cost': self.config['resource_cost'],
                'accuracy_estimate': 0.0,
                'success': False
            }

    def _generate_simple_response(self, query):
        """Generate a simple response based on query"""
        query_lower = query.lower()

        # Basic keyword-based responses
        if any(word in query_lower for word in ['what', 'who', 'when', 'where']):
            return f"Based on your query about '{query[:50]}...', here is a basic answer. This is a simple response from the lightweight model."

        elif any(word in query_lower for word in ['yes', 'no', 'is', 'are']):
            return "Yes, based on general knowledge. This is a quick response from the simple model."

        else:
            return f"I understand you're asking about '{query[:30]}...'. Here's a brief response from the simple model."


class MediumModel(BaseModel):
    """Medium complexity model for balanced performance"""

    def __init__(self):
        super().__init__('medium')

    def process_query(self, query):
        """Process query with medium model logic"""
        start_time = time.time()

        try:
            # Simulate processing time
            processing_time = random.uniform(1.5, 2.5)  # 1.5-2.5 seconds
            time.sleep(processing_time)

            # Medium complexity response generation
            response = self._generate_medium_response(query)

            # Calculate metrics
            response_time = time.time() - start_time
            resource_cost = self.config['resource_cost']

            # Update statistics
            self._update_stats(True, response_time, resource_cost)

            return {
                'response': response,
                'model_used': self.name,
                'response_time': response_time,
                'resource_cost': resource_cost,
                'accuracy_estimate': self.config['accuracy_score'],
                'success': True
            }

        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time, self.config['resource_cost'])

            return {
                'response': f"Error processing query: {str(e)}",
                'model_used': self.name,
                'response_time': response_time,
                'resource_cost': self.config['resource_cost'],
                'accuracy_estimate': 0.0,
                'success': False
            }

    def _generate_medium_response(self, query):
        """Generate a medium complexity response"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['how', 'why', 'explain']):
            return f"To address your question about '{query[:50]}...', let me provide a detailed explanation. This medium model can handle moderately complex reasoning and provide structured responses with multiple points and considerations."

        elif any(word in query_lower for word in ['compare', 'difference', 'between']):
            return f"Comparing the elements in your query '{query[:50]}...', there are several key differences and similarities. This balanced model can analyze multiple aspects and provide comparative insights."

        else:
            return f"Regarding your query '{query[:40]}...', I can provide a comprehensive response. This medium model offers balanced performance with good accuracy and reasonable response time."


class AdvancedModel(BaseModel):
    """Advanced model for complex analysis and reasoning"""

    def __init__(self):
        super().__init__('advanced')

    def process_query(self, query):
        """Process query with advanced model logic"""
        start_time = time.time()

        try:
            # Simulate processing time
            processing_time = random.uniform(6.0, 10.0)  # 6-10 seconds
            time.sleep(processing_time)

            # Advanced response generation
            response = self._generate_advanced_response(query)

            # Calculate metrics
            response_time = time.time() - start_time
            resource_cost = self.config['resource_cost']

            # Update statistics
            self._update_stats(True, response_time, resource_cost)

            return {
                'response': response,
                'model_used': self.name,
                'response_time': response_time,
                'resource_cost': resource_cost,
                'accuracy_estimate': self.config['accuracy_score'],
                'success': True
            }

        except Exception as e:
            response_time = time.time() - start_time
            self._update_stats(False, response_time, resource_cost)

            return {
                'response': f"Error processing query: {str(e)}",
                'model_used': self.name,
                'response_time': response_time,
                'resource_cost': self.config['resource_cost'],
                'accuracy_estimate': 0.0,
                'success': False
            }

    def _generate_advanced_response(self, query):
        """Generate an advanced, detailed response"""
        query_lower = query.lower()

        if any(word in query_lower for word in ['analyze', 'evaluate', 'design']):
            return f"Conducting a comprehensive analysis of '{query[:50]}...': This advanced model performs deep reasoning, considering multiple perspectives, potential implications, edge cases, and providing detailed recommendations. The analysis includes systematic evaluation of alternatives, risk assessment, and strategic considerations."

        elif any(word in query_lower for word in ['algorithm', 'strategy', 'complex']):
            return f"For the complex problem '{query[:50]}...', I'll provide a sophisticated solution. This advanced model can handle intricate logical reasoning, algorithm design, strategic planning, and complex problem-solving with high accuracy and detailed step-by-step analysis."

        else:
            return f"Providing comprehensive analysis for '{query[:40]}...': This high-capability model offers detailed reasoning, multiple solution approaches, thorough evaluation of options, and deep insights. The response includes nuanced understanding and sophisticated problem-solving capabilities."


class ModelFactory:
    """Factory class for creating model instances"""

    _models = {
        'simple': SimpleModel,
        'medium': MediumModel,
        'advanced': AdvancedModel
    }

    @classmethod
    def create_model(cls, model_name):
        """Create and return a model instance"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls._models.keys())}")

        return cls._models[model_name]()

    @classmethod
    def get_available_models(cls):
        """Get list of available model names"""
        return list(cls._models.keys())

    @classmethod
    def create_all_models(cls):
        """Create instances of all available models"""
        return {name: cls.create_model(name) for name in cls._models.keys()}


# Utility function for model comparison
def compare_models(query, models=None):
    """Compare performance of different models on the same query"""
    if models is None:
        models = ModelFactory.create_all_models()

    results = {}
    for model_name, model in models.items():
        result = model.process_query(query)
        results[model_name] = result

    return results