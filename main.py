# main.py

import sys
import time
import argparse
from typing import Optional

# Ensure console supports UTF-8, especially on Windows
if sys.platform == "win32":
    try:
        import os
        os.system("chcp 65001 > nul")
    except Exception as e:
        print(f"Could not set console to UTF-8. Error: {e}")

from routing_system import DynamicRoutingSystem, create_routing_system
from evaluator import SystemEvaluator
from test_queries import TestQueryManager, quick_test_queries, comprehensive_test_queries

class DynamicRoutingCLI:
    """Command Line Interface for the Dynamic Routing System."""
    def __init__(self):
        self.routing_system: Optional[DynamicRoutingSystem] = None
        self.evaluator: Optional[SystemEvaluator] = None
        self.query_manager = TestQueryManager()

    def initialize_system(self):
        """Initializes all necessary system components."""
        print("🚀 Initializing Dynamic Routing System...")
        try:
            self.routing_system = create_routing_system()
            self.evaluator = SystemEvaluator()
            print("✅ System initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False

    def run_interactive_mode(self):
        """Runs the application in a live interactive mode."""
        print("\n🔄 Interactive Query Mode (type 'exit' to quit)")
        print("Commands: stats, cache, help")
        print("-" * 50)
        while True:
            try:
                user_input = input("\nQuery> ").strip()
                if not user_input: continue

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'stats':
                    self._show_stats()
                elif user_input.lower() == 'cache':
                    self._show_cache_info()
                else:
                    self._process_interactive_query(user_input)

            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")

    def _process_interactive_query(self, query: str):
        """Processes a single query and prints the formatted result."""
        print(f"\n📝 Processing: '{query[:60]}...'")
        result = self.routing_system.process_query(query)

        if result.get('success'):
            info = result.get('routing_info', {})
            print("✅ Success!")
            print(f"  - Model Used: {info.get('actual_model_used', 'N/A')}")
            print(f"  - Cache Hit: {'Yes' if info.get('cache_hit') else 'No'}")
            print(f"  - Fallback Triggered: {'Yes' if info.get('fallback_triggered') else 'No'}")
            print(f"  - Processing Time: {info.get('processing_time', 0):.3f}s")

            if info.get('classification'):
                class_info = info['classification']
                print(f"  - Recommendation: '{class_info.get('recommended_model', 'N/A')}' (Confidence: {class_info.get('confidence', 0):.1%})")

            print(f"\n📄 Response:\n{result.get('response', '')}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

    def _show_stats(self):
        """Displays a summary of system statistics."""
        stats = self.routing_system.get_detailed_stats()
        print("\n--- System Stats ---")
        for key, value in stats.get('system', {}).items():
            print(f"{str(key).replace('_', ' ').title()}: {value}")

    def _show_cache_info(self):
        """Displays a summary of the cache contents."""
        info = self.routing_system.cache_manager.get_cache_info()
        print("\n--- Cache Info ---")
        print(f"Total entries: {info['total_entries']}")
        print("Top 3 most accessed:")
        for entry in info['entries'][:3]:
            print(f"  - '{entry['query'][:40]}...' (Accessed {entry['access_count']} times)")

    def _show_help(self):
        """Displays help information for interactive commands."""
        print("\nAvailable interactive commands: stats, cache, help, exit")

    def run_evaluation(self, suite_name: str):
        """Runs a specified evaluation suite."""
        print(f"\n🔍 Running '{suite_name}' evaluation...")
        if suite_name == 'quick':
            queries = quick_test_queries()
        else:
            queries = comprehensive_test_queries()

        if not queries:
            print(f"❌ Unknown or empty test suite: {suite_name}")
            return

        self.evaluator.evaluate_system(queries)

    def shutdown(self):
        """Shuts down the system gracefully."""
        if self.routing_system:
            self.routing_system.shutdown()

def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Dynamic Routing System for LLM Queries.")
    parser.add_argument('-i', '--interactive', action='store_true', help="Run in interactive mode.")
    parser.add_argument('-e', '--evaluate', choices=['quick', 'comprehensive'], help="Run a system evaluation suite.")
    parser.add_argument('-q', '--query', type=str, help="Process a single query and exit.")

    args = parser.parse_args()

    print("=" * 60)
    print("🤖 DYNAMIC ROUTING SYSTEM FOR LLM QUERIES")
    print("=" * 60)

    cli = DynamicRoutingCLI()
    if not cli.initialize_system():
        sys.exit(1)

    try:
        if args.interactive:
            cli.run_interactive_mode()
        elif args.evaluate:
            cli.run_evaluation(args.evaluate)
        elif args.query:
            cli._process_interactive_query(args.query)
        else:
            # Default action if no arguments are provided
            print("No mode selected. Running quick evaluation by default.")
            cli.run_evaluation('quick')
            print("\n💡 Tip: Use 'python main.py --interactive' for live queries.")

    except KeyboardInterrupt:
        print("\n\n👋 User interrupted. Shutting down...")
    finally:
        cli.shutdown()

if __name__ == "__main__":
    main()