"""
System Evaluator for Dynamic Routing System
Evaluates system performance, compares routing strategies, and analyzes effectiveness
"""

import time
import json
import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from routing_system import DynamicRoutingSystem, create_routing_system
from models import ModelFactory, compare_models
from config import get_all_model_names, FILE_PATHS, MODEL_CONFIGS


class QueryResult:
    """Represents the result of processing a single query"""

    def __init__(self, query: str, expected_complexity: str = 'medium'):
        self.query = query
        self.expected_complexity = expected_complexity
        self.routing_result = None
        self.direct_results = {}  # Results from testing each model directly
        self.evaluation_metrics = {}
        self.timestamp = time.time()

    def add_routing_result(self, result: Dict[str, Any]):
        """Add result from dynamic routing system"""
        self.routing_result = result

    def add_direct_result(self, model_name: str, result: Dict[str, Any]):
        """Add result from direct model testing"""
        self.direct_results[model_name] = result

    def calculate_metrics(self):
        """Calculate evaluation metrics for this query"""
        if not self.routing_result or not self.direct_results:
            return

        routing_info = self.routing_result.get('routing_info', {})

        # Basic metrics
        self.evaluation_metrics = {
            'query_length': len(self.query),
            'expected_complexity': self.expected_complexity,
            'recommended_model': routing_info.get('recommended_model', 'unknown'),
            'actual_model_used': routing_info.get('actual_model_used', 'unknown'),
            'routing_success': self.routing_result.get('success', False),
            'cache_hit': routing_info.get('cache_hit', False),
            'fallback_triggered': routing_info.get('fallback_triggered', False),
            'processing_time': routing_info.get('processing_time', 0),
            'total_time': routing_info.get('total_time', 0),
            'resource_cost': routing_info.get('resource_cost', 0)
        }

        # Classification accuracy (if expected complexity provided)
        recommended = self.evaluation_metrics['recommended_model']
        expected = self.expected_complexity
        self.evaluation_metrics['classification_correct'] = (recommended == expected)

        # Efficiency metrics (compare with direct model usage)
        self._calculate_efficiency_metrics()

    def _calculate_efficiency_metrics(self):
        """Calculate efficiency compared to always using best/worst model"""
        if not self.direct_results:
            return

        routing_time = self.evaluation_metrics['processing_time']
        routing_cost = self.evaluation_metrics['resource_cost']

        # Compare with always using advanced model
        if 'advanced' in self.direct_results:
            advanced_time = self.direct_results['advanced'].get('response_time', 0)
            advanced_cost = self.direct_results['advanced'].get('resource_cost', 0)

            self.evaluation_metrics['time_savings_vs_advanced'] = advanced_time - routing_time
            self.evaluation_metrics['cost_savings_vs_advanced'] = advanced_cost - routing_cost
            self.evaluation_metrics['efficiency_vs_advanced'] = (
                                                                        (advanced_time + advanced_cost) - (routing_time + routing_cost)
                                                                ) / max(advanced_time + advanced_cost, 1)

        # Compare with always using simple model
        if 'simple' in self.direct_results:
            simple_time = self.direct_results['simple'].get('response_time', 0)
            simple_cost = self.direct_results['simple'].get('resource_cost', 0)

            self.evaluation_metrics['time_overhead_vs_simple'] = routing_time - simple_time
            self.evaluation_metrics['cost_overhead_vs_simple'] = routing_cost - simple_cost


class SystemEvaluator:
    """Main system evaluator for comprehensive performance analysis"""

    def __init__(self):
        self.routing_system = None
        self.all_models = ModelFactory.create_all_models()
        self.evaluation_results: List[QueryResult] = []

        # Evaluation settings
        self.evaluation_config = {
            'include_direct_comparison': True,
            'include_cache_analysis': True,
            'include_fallback_testing': True,
            'save_detailed_results': True,
            'generate_visualizations': True
        }

    def evaluate_system(self, test_queries: List[Dict[str, Any]],
                        reset_system: bool = True) -> Dict[str, Any]:
        """
        Comprehensive system evaluation

        Args:
            test_queries: List of queries with expected complexity levels
            reset_system: Whether to reset system stats before evaluation

        Returns:
            Dictionary containing evaluation results and metrics
        """
        print("Starting comprehensive system evaluation...")

        # Initialize fresh routing system
        self.routing_system = create_routing_system()
        if reset_system:
            self.routing_system.reset_stats()

        # Clear previous results
        self.evaluation_results.clear()

        # Process each query
        for i, query_data in enumerate(test_queries):
            print(f"Evaluating query {i+1}/{len(test_queries)}: {query_data['query'][:50]}...")

            query_result = self._evaluate_single_query(
                query_data['query'],
                query_data.get('expected_complexity', 'medium')
            )
            self.evaluation_results.append(query_result)

        # Generate comprehensive analysis
        analysis = self._generate_comprehensive_analysis()

        # Save results if configured
        if self.evaluation_config['save_detailed_results']:
            self._save_evaluation_results(analysis)

        # Generate visualizations
        if self.evaluation_config['generate_visualizations']:
            self._generate_visualizations()

        print("System evaluation completed!")
        return analysis

    def _evaluate_single_query(self, query: str, expected_complexity: str) -> QueryResult:
        """Evaluate a single query comprehensively"""
        query_result = QueryResult(query, expected_complexity)

        # Test with dynamic routing system
        routing_result = self.routing_system.process_query(query)
        query_result.add_routing_result(routing_result)

        # Test with direct model comparisons (if enabled)
        if self.evaluation_config['include_direct_comparison']:
            direct_results = compare_models(query, self.all_models)
            for model_name, result in direct_results.items():
                query_result.add_direct_result(model_name, result)

        # Calculate metrics
        query_result.calculate_metrics()

        return query_result

    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of evaluation results"""
        if not self.evaluation_results:
            return {'error': 'No evaluation results available'}

        analysis = {
            'evaluation_summary': self._generate_summary(),
            'performance_metrics': self._calculate_performance_metrics(),
            'classification_analysis': self._analyze_classification_accuracy(),
            'efficiency_analysis': self._analyze_efficiency(),
            'caching_analysis': self._analyze_caching_effectiveness(),
            'fallback_analysis': self._analyze_fallback_effectiveness(),
            'model_usage_analysis': self._analyze_model_usage(),
            'recommendations': self._generate_recommendations()
        }

        return analysis

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary statistics"""
        total_queries = len(self.evaluation_results)
        successful_queries = sum(1 for r in self.evaluation_results
                                 if r.routing_result and r.routing_result.get('success', False))

        cache_hits = sum(1 for r in self.evaluation_results
                         if r.evaluation_metrics.get('cache_hit', False))

        fallback_triggers = sum(1 for r in self.evaluation_results
                                if r.evaluation_metrics.get('fallback_triggered', False))

        total_time = sum(r.evaluation_metrics.get('total_time', 0) for r in self.evaluation_results)
        total_cost = sum(r.evaluation_metrics.get('resource_cost', 0) for r in self.evaluation_results)

        return {
            'total_queries_evaluated': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / max(total_queries, 1),
            'cache_hits': cache_hits,
            'cache_hit_rate': cache_hits / max(total_queries, 1),
            'fallback_triggers': fallback_triggers,
            'fallback_rate': fallback_triggers / max(total_queries, 1),
            'total_processing_time': total_time,
            'average_processing_time': total_time / max(total_queries, 1),
            'total_resource_cost': total_cost,
            'average_resource_cost': total_cost / max(total_queries, 1)
        }

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate detailed performance metrics"""
        processing_times = [r.evaluation_metrics.get('processing_time', 0)
                            for r in self.evaluation_results]
        resource_costs = [r.evaluation_metrics.get('resource_cost', 0)
                          for r in self.evaluation_results]

        return {
            'processing_time_stats': {
                'mean': statistics.mean(processing_times) if processing_times else 0,
                'median': statistics.median(processing_times) if processing_times else 0,
                'std_dev': statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0
            },
            'resource_cost_stats': {
                'mean': statistics.mean(resource_costs) if resource_costs else 0,
                'median': statistics.median(resource_costs) if resource_costs else 0,
                'std_dev': statistics.stdev(resource_costs) if len(resource_costs) > 1 else 0,
                'min': min(resource_costs) if resource_costs else 0,
                'max': max(resource_costs) if resource_costs else 0
            }
        }

    def _analyze_classification_accuracy(self) -> Dict[str, Any]:
        """Analyze query classification accuracy"""
        correct_classifications = sum(1 for r in self.evaluation_results
                                      if r.evaluation_metrics.get('classification_correct', False))

        total_with_expected = sum(1 for r in self.evaluation_results
                                  if r.expected_complexity != 'medium' or
                                  r.evaluation_metrics.get('recommended_model') != 'medium')

        # Classification breakdown by complexity
        classification_breakdown = {}
        for complexity in ['simple', 'medium', 'advanced']:
            expected_count = sum(1 for r in self.evaluation_results
                                 if r.expected_complexity == complexity)
            correct_count = sum(1 for r in self.evaluation_results
                                if r.expected_complexity == complexity and
                                r.evaluation_metrics.get('classification_correct', False))

            classification_breakdown[complexity] = {
                'expected_count': expected_count,
                'correct_count': correct_count,
                'accuracy': correct_count / max(expected_count, 1)
            }

        return {
            'overall_accuracy': correct_classifications / max(len(self.evaluation_results), 1),
            'classification_breakdown': classification_breakdown,
            'confusion_matrix': self._generate_confusion_matrix()
        }

    def _generate_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix for classification accuracy"""
        matrix = {
            'simple': {'simple': 0, 'medium': 0, 'advanced': 0},
            'medium': {'simple': 0, 'medium': 0, 'advanced': 0},
            'advanced': {'simple': 0, 'medium': 0, 'advanced': 0}
        }

        for result in self.evaluation_results:
            expected = result.expected_complexity
            predicted = result.evaluation_metrics.get('recommended_model', 'medium')

            if expected in matrix and predicted in matrix[expected]:
                matrix[expected][predicted] += 1

        return matrix

    def _analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze system efficiency compared to baseline strategies"""
        efficiency_metrics = {
            'vs_always_advanced': [],
            'vs_always_simple': []
        }

        for result in self.evaluation_results:
            if 'efficiency_vs_advanced' in result.evaluation_metrics:
                efficiency_metrics['vs_always_advanced'].append(
                    result.evaluation_metrics['efficiency_vs_advanced']
                )

            if 'time_overhead_vs_simple' in result.evaluation_metrics:
                simple_overhead = result.evaluation_metrics['time_overhead_vs_simple']
                cost_overhead = result.evaluation_metrics.get('cost_overhead_vs_simple', 0)
                efficiency_metrics['vs_always_simple'].append(simple_overhead + cost_overhead)

        analysis = {}

        if efficiency_metrics['vs_always_advanced']:
            adv_efficiencies = efficiency_metrics['vs_always_advanced']
            analysis['vs_always_advanced'] = {
                'average_efficiency_gain': statistics.mean(adv_efficiencies),
                'efficiency_gain_std': statistics.stdev(adv_efficiencies) if len(adv_efficiencies) > 1 else 0,
                'queries_more_efficient': sum(1 for e in adv_efficiencies if e > 0),
                'efficiency_rate': sum(1 for e in adv_efficiencies if e > 0) / len(adv_efficiencies)
            }

        return analysis

    def _analyze_caching_effectiveness(self) -> Dict[str, Any]:
        """Analyze caching system effectiveness"""
        cache_stats = self.routing_system.cache_manager.get_stats()

        return {
            'cache_performance': cache_stats,
            'cache_impact_analysis': {
                'queries_with_cache_hits': sum(1 for r in self.evaluation_results
                                               if r.evaluation_metrics.get('cache_hit', False)),
                'average_time_saved': self._calculate_average_cache_time_savings(),
                'resource_cost_saved': sum(r.evaluation_metrics.get('resource_cost', 0)
                                           for r in self.evaluation_results
                                           if r.evaluation_metrics.get('cache_hit', False))
            }
        }

    def _calculate_average_cache_time_savings(self) -> float:
        """Calculate average time savings from cache hits"""
        # This is a simplified calculation
        # In practice, you'd need to know the original processing time
        cache_hit_times = [r.evaluation_metrics.get('processing_time', 0)
                           for r in self.evaluation_results
                           if r.evaluation_metrics.get('cache_hit', False)]

        if not cache_hit_times:
            return 0.0

        # Assume cache hits save significant time compared to processing
        average_processing_time = statistics.mean([
            r.evaluation_metrics.get('processing_time', 0)
            for r in self.evaluation_results
            if not r.evaluation_metrics.get('cache_hit', False)
        ]) if self.evaluation_results else 1.0

        return average_processing_time * 0.9  # Assume 90% time savings

    def _analyze_fallback_effectiveness(self) -> Dict[str, Any]:
        """Analyze fallback strategy effectiveness"""
        fallback_queries = [r for r in self.evaluation_results
                            if r.evaluation_metrics.get('fallback_triggered', False)]

        if not fallback_queries:
            return {'no_fallbacks_triggered': True}

        return {
            'fallback_success_rate': sum(1 for r in fallback_queries
                                         if r.routing_result.get('success', False)) / len(fallback_queries),
            'common_fallback_scenarios': self._identify_common_fallback_scenarios(fallback_queries),
            'fallback_performance_impact': self._calculate_fallback_performance_impact(fallback_queries)
        }

    def _identify_common_fallback_scenarios(self, fallback_queries: List[QueryResult]) -> Dict[str, int]:
        """Identify common scenarios that trigger fallbacks"""
        scenarios = {}

        for result in fallback_queries:
            # Analyze query characteristics
            length = result.evaluation_metrics['query_length']
            complexity = result.expected_complexity

            scenario = f"{complexity}_query"
            if length > 200:
                scenario += "_long"
            elif length < 50:
                scenario += "_short"

            scenarios[scenario] = scenarios.get(scenario, 0) + 1

        return scenarios

    def _calculate_fallback_performance_impact(self, fallback_queries: List[QueryResult]) -> Dict[str, float]:
        """Calculate performance impact of fallback usage"""
        fallback_times = [r.evaluation_metrics.get('total_time', 0) for r in fallback_queries]
        normal_times = [r.evaluation_metrics.get('total_time', 0) for r in self.evaluation_results
                        if not r.evaluation_metrics.get('fallback_triggered', False)]

        return {
            'average_fallback_time': statistics.mean(fallback_times) if fallback_times else 0,
            'average_normal_time': statistics.mean(normal_times) if normal_times else 0,
            'fallback_overhead': (statistics.mean(fallback_times) - statistics.mean(normal_times))
            if fallback_times and normal_times else 0
        }

    def _analyze_model_usage(self) -> Dict[str, Any]:
        """Analyze how different models were used"""
        model_usage = {}

        for result in self.evaluation_results:
            model_used = result.evaluation_metrics.get('actual_model_used', 'unknown')
            if model_used not in model_usage:
                model_usage[model_used] = {
                    'count': 0,
                    'avg_processing_time': 0,
                    'avg_resource_cost': 0,
                    'success_rate': 0
                }

            model_usage[model_used]['count'] += 1
            model_usage[model_used]['avg_processing_time'] += result.evaluation_metrics.get('processing_time', 0)
            model_usage[model_used]['avg_resource_cost'] += result.evaluation_metrics.get('resource_cost', 0)
            if result.routing_result and result.routing_result.get('success', False):
                model_usage[model_used]['success_rate'] += 1

        # Calculate averages
        for model, stats in model_usage.items():
            count = max(stats['count'], 1)
            stats['avg_processing_time'] /= count
            stats['avg_resource_cost'] /= count
            stats['success_rate'] /= count
            stats['usage_percentage'] = stats['count'] / len(self.evaluation_results)

        return model_usage

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []

        # Check classification accuracy
        classification_analysis = self._analyze_classification_accuracy()
        overall_accuracy = classification_analysis['overall_accuracy']

        if overall_accuracy < 0.8:
            recommendations.append(
                f"Classification accuracy is {overall_accuracy:.2%}. Consider adjusting classification criteria."
            )

        # Check cache effectiveness
        cache_stats = self.routing_system.cache_manager.get_stats()
        hit_rate = cache_stats.get('hit_rate', 0)

        if hit_rate < 0.3:
            recommendations.append(
                f"Cache hit rate is {hit_rate:.2%}. Consider increasing cache size or adjusting similarity threshold."
            )

        # Check model usage balance
        model_usage = self._analyze_model_usage()
        advanced_usage = model_usage.get('advanced', {}).get('usage_percentage', 0)

        if advanced_usage > 0.5:
            recommendations.append(
                f"Advanced model used {advanced_usage:.2%} of the time. Consider adjusting classification to use lighter models."
            )

        # Check fallback frequency
        summary = self._generate_summary()
        fallback_rate = summary['fallback_rate']

        if fallback_rate > 0.2:
            recommendations.append(
                f"Fallback triggered {fallback_rate:.2%} of queries. Consider reviewing timeout thresholds or model reliability."
            )

        if not recommendations:
            recommendations.append("System performance is within expected parameters.")

        return recommendations

    def _save_evaluation_results(self, analysis: Dict[str, Any]):
        """Save evaluation results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

            # Prepare data for saving
            save_data = {
                'evaluation_timestamp': timestamp,
                'analysis': analysis,
                'detailed_results': [
                    {
                        'query': result.query,
                        'expected_complexity': result.expected_complexity,
                        'metrics': result.evaluation_metrics,
                        'routing_result': result.routing_result
                    }
                    for result in self.evaluation_results
                ],
                'system_configuration': {
                    'models': get_all_model_names(),
                    'model_configs': MODEL_CONFIGS
                }
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

            print(f"Evaluation results saved to {filename}")

        except Exception as e:
            print(f"Error saving evaluation results: {e}")

    def _generate_visualizations(self):
        """Generate visualization plots for evaluation results"""
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Dynamic Routing System Evaluation Results', fontsize=16)

            # 1. Model Usage Distribution
            model_usage = self._analyze_model_usage()
            models = list(model_usage.keys())
            usage_counts = [model_usage[m]['count'] for m in models]

            axes[0, 0].pie(usage_counts, labels=models, autopct='%1.1f%%')
            axes[0, 0].set_title('Model Usage Distribution')

            # 2. Processing Time Distribution
            processing_times = [r.evaluation_metrics.get('processing_time', 0)
                                for r in self.evaluation_results]
            axes[0, 1].hist(processing_times, bins=10, alpha=0.7)
            axes[0, 1].set_title('Processing Time Distribution')
            axes[0, 1].set_xlabel('Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')

            # 3. Resource Cost by Model
            model_costs = {}
            for result in self.evaluation_results:
                model = result.evaluation_metrics.get('actual_model_used', 'unknown')
                cost = result.evaluation_metrics.get('resource_cost', 0)
                if model not in model_costs:
                    model_costs[model] = []
                model_costs[model].append(cost)

            models = list(model_costs.keys())
            costs = [model_costs[m] for m in models]
            axes[0, 2].boxplot(costs, labels=models)
            axes[0, 2].set_title('Resource Cost by Model')
            axes[0, 2].set_ylabel('Resource Cost')

            # 4. Classification Accuracy
            confusion_matrix = self._generate_confusion_matrix()
            confusion_df = pd.DataFrame(confusion_matrix)
            sns.heatmap(confusion_df, annot=True, fmt='d', ax=axes[1, 0])
            axes[1, 0].set_title('Classification Confusion Matrix')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')

            # 5. Cache Performance
            cache_results = ['Cache Hit' if r.evaluation_metrics.get('cache_hit', False)
                             else 'Cache Miss' for r in self.evaluation_results]
            cache_counts = pd.Series(cache_results).value_counts()
            axes[1, 1].bar(cache_counts.index, cache_counts.values)
            axes[1, 1].set_title('Cache Performance')
            axes[1, 1].set_ylabel('Count')

            # 6. Success Rate by Query Length
            lengths = [r.evaluation_metrics.get('query_length', 0) for r in self.evaluation_results]
            successes = [1 if r.routing_result and r.routing_result.get('success', False) else 0
                         for r in self.evaluation_results]

            axes[1, 2].scatter(lengths, successes, alpha=0.6)
            axes[1, 2].set_title('Success Rate vs Query Length')
            axes[1, 2].set_xlabel('Query Length (characters)')
            axes[1, 2].set_ylabel('Success (0/1)')

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"evaluation_plots_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"Visualization saved to {plot_filename}")

        except Exception as e:
            print(f"Error generating visualizations: {e}")


def quick_evaluate(test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quick evaluation function for convenience"""
    evaluator = SystemEvaluator()
    return evaluator.evaluate_system(test_queries)


if __name__ == "__main__":
    print("Testing System Evaluator...")

    # Sample test queries for demonstration
    sample_queries = [
        {'query': 'What is Python?', 'expected_complexity': 'simple'},
        {'query': 'How do I implement a binary search algorithm in Python?', 'expected_complexity': 'medium'},
        {'query': 'Analyze the computational complexity of different machine learning algorithms and design an optimal distributed system architecture', 'expected_complexity': 'advanced'},
        {'query': 'What time is it?', 'expected_complexity': 'simple'},
        {'query': 'Explain the differences between supervised and unsupervised learning', 'expected_complexity': 'medium'}
    ]

    # Run evaluation
    evaluator = SystemEvaluator()
    results = evaluator.evaluate_system(sample_queries)

    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    summary = results['evaluation_summary']
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print("\n=== RECOMMENDATIONS ===")
    for recommendation in results['recommendations']:
        print(f"• {recommendation}")