# evaluator.py (النسخة النهائية والمصلحة)

import time
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from routing_system import create_routing_system

class SystemEvaluator:
    """Automates the process of evaluating the routing system's performance."""
    def __init__(self):
        self.routing_system = create_routing_system()
        self.evaluation_results = []

    def evaluate_system(self, test_queries: List[Dict[str, Any]]) -> Dict:
        """Runs the evaluation and returns a summary."""
        print(f"--- Starting New Evaluation with {len(test_queries)} queries ---")
        self.evaluation_results.clear()

        print("--> Clearing cache to ensure fresh results...")
        if hasattr(self.routing_system, 'cache_manager') and hasattr(self.routing_system.cache_manager, 'clear_cache'):
            cleared_count = self.routing_system.cache_manager.clear_cache()
            print(f"--> Cache cleared. {cleared_count} items removed.")

        print("--> Processing queries...")
        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            if not query: continue

            result = self.routing_system.process_query(query)

            # Combine base result with routing_info for a flat structure
            record = {
                'query': query,
                'expected_complexity': query_data.get('expected_complexity'),
                'success': result.get('success', False),
                'response': result.get('response', ''),
                **result.get('routing_info', {})
            }
            self.evaluation_results.append(record)
            time.sleep(0.1)

        print("--> Analyzing results...")
        analysis = self._analyze_results()
        self._generate_visualizations()
        self._save_results_to_file(analysis)

        return analysis

    def _analyze_results(self) -> Dict:
        """Analyzes the collected results using pandas."""
        if not self.evaluation_results: return {}

        df = pd.DataFrame(self.evaluation_results)

        def check_classification(row):
            if isinstance(row.get('classification'), dict):
                return row.get('expected_complexity') == row.get('classification', {}).get('recommended_model')
            return None
        df['classification_correct'] = df.apply(check_classification, axis=1)

        summary = {
            'total_queries': len(df),
            'success_rate': df['success'].mean(),
            'cache_hit_rate': df['cache_hit'].mean(),
            'fallback_rate': df['fallback_triggered'].mean(),
            'avg_processing_time': df[df['processing_time'] > 0]['processing_time'].mean(),
            'classification_accuracy': df['classification_correct'].mean()
        }

        print("\n--- Evaluation Summary ---")
        for key, val in summary.items():
            is_rate = 'rate' in key or 'accuracy' in key
            if is_rate and pd.notna(val):
                formatted_val = f"{val:.2%}"
            else:
                formatted_val = f"{val:.3f}" if isinstance(val, float) else val
            print(f"{key.replace('_', ' ').title()}: {formatted_val}")

        return summary

    def _save_results_to_file(self, analysis: Dict):
        """Saves detailed results to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        data_to_save = {
            'summary': analysis,
            'details': self.evaluation_results
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        print(f"\nFull evaluation results saved to {filename}")

    def _generate_visualizations(self):
        """Generates and saves performance plots."""
        if not self.evaluation_results: return

        df = pd.DataFrame(self.evaluation_results)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('System Performance Dashboard', fontsize=18)

        # Plot 1: Model Usage Distribution
        sns.countplot(ax=axes[0, 0], data=df, x='actual_model_used', order=['simple', 'medium', 'advanced'])
        axes[0, 0].set_title('Model Usage Distribution')

        # Plot 2: Processing Time by Model
        sns.boxplot(ax=axes[0, 1], data=df, x='actual_model_used', y='processing_time', order=['simple', 'medium', 'advanced'])
        axes[0, 1].set_title('Processing Time by Model')

        # Plot 3: Cache Hit vs. Miss Rate
        cache_counts = df['cache_hit'].value_counts()
        if not cache_counts.empty:
            axes[1, 0].pie(cache_counts, labels=cache_counts.index.map({True: 'Hit', False: 'Miss'}), autopct='%1.1f%%')
        axes[1, 0].set_title('Cache Hit vs. Miss Rate')

        # Plot 4: Classification Accuracy
        df['classification_correct'] = df.apply(
            lambda row: row.get('expected_complexity') == row.get('classification', {}).get('recommended_model') if isinstance(row.get('classification'), dict) else None,
            axis=1
        )
        if df['classification_correct'].notna().any():
            accuracy_counts = df['classification_correct'].value_counts()
            axes[1, 1].pie(accuracy_counts, labels=accuracy_counts.index.map({True: 'Correct', False: 'Incorrect'}), autopct='%1.1f%%', colors=['#4caf50', '#f44336'])
        else:
            axes[1, 1].text(0.5, 0.5, 'No classification data available', ha='center', va='center')
        axes[1, 1].set_title('Classification Accuracy')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()