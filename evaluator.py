# evaluator.py

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
        print(f"Starting evaluation with {len(test_queries)} queries...")
        self.evaluation_results.clear()

        for i, query_data in enumerate(test_queries):
            query = query_data.get('query', '')
            print(f"  ({i+1}/{len(test_queries)}) Evaluating: '{query[:40]}...'")

            result = self.routing_system.process_query(query)

            record = {
                'query': query,
                'expected_complexity': query_data.get('expected_complexity'),
                'success': result.get('success', False),
                **result.get('routing_info', {})
            }
            self.evaluation_results.append(record)
            time.sleep(0.1) # Prevents overwhelming the system in a real scenario

        analysis = self._analyze_results()
        self._save_results_to_file(analysis)
        self._generate_visualizations()

        return analysis

    def _analyze_results(self) -> Dict:
        """Analyzes the collected results using pandas."""
        if not self.evaluation_results:
            return {'error': 'No results to analyze.'}

        df = pd.DataFrame(self.evaluation_results)

        def check_classification(row):
            if isinstance(row.get('classification'), dict):
                return row['expected_complexity'] == row['classification'].get('recommended_model')
            return None

        df['classification_correct'] = df.apply(check_classification, axis=1)

        summary = {
            'total_queries': len(df),
            'success_rate': df['success'].mean(),
            'cache_hit_rate': df['cache_hit'].mean(),
            'fallback_rate': df['fallback_triggered'].mean(),
            'avg_processing_time': df[df['processing_time'] > 0]['processing_time'].mean(),
            'avg_resource_cost': df['resource_cost'].mean(),
            'classification_accuracy': df['classification_correct'].mean()
        }

        print("\n--- Evaluation Summary ---")
        for key, val in summary.items():
            is_rate = 'rate' in key or 'accuracy' in key
            print(f"{key.replace('_', ' ').title()}: {val:.2% if is_rate and pd.notna(val) else val}")

        return summary

    def _save_results_to_file(self, analysis: Dict):
        """Saves detailed results to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        data_to_save = {'summary': analysis, 'details': self.evaluation_results}

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

        sns.countplot(ax=axes[0, 0], data=df, x='actual_model_used', order=['simple', 'medium', 'advanced'], palette='viridis')
        axes[0, 0].set_title('Model Usage Distribution')

        sns.boxplot(ax=axes[0, 1], data=df, x='actual_model_used', y='processing_time', order=['simple', 'medium', 'advanced'], palette='plasma')
        axes[0, 1].set_title('Processing Time by Model')

        cache_counts = df['cache_hit'].value_counts()
        axes[1, 0].pie(cache_counts, labels=cache_counts.index.map({True: 'Hit', False: 'Miss'}), autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Cache Hit vs. Miss Rate')

        if 'classification_correct' in df.columns:
            accuracy_counts = df['classification_correct'].value_counts()
            axes[1, 1].pie(accuracy_counts, labels=accuracy_counts.index.map({True: 'Correct', False: 'Incorrect'}), autopct='%1.1f%%', colors=['#4caf50', '#f44336'])
            axes[1, 1].set_title('Classification Accuracy')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_filename)
        print(f"Visualizations saved to {plot_filename}")
        plt.show()