"""
Dynamic Routing System - Main Application
Entry point for the LLM Dynamic Routing System with CLI interface
"""

import sys
import time
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import all system components
from routing_system import DynamicRoutingSystem, create_routing_system, quick_query
from evaluator import SystemEvaluator, quick_evaluate
from test_queries import (
    TestQueryManager, quick_test_queries, comprehensive_test_queries,
    COMPREHENSIVE_TEST_SUITE
)
from query_classifier import classify_single_query
from models import ModelFactory, compare_models
from cache_manager import CacheManager
from config import get_all_model_names, GENERAL_SETTINGS


class DynamicRoutingCLI:
    """Command Line Interface for the Dynamic Routing System"""

    def __init__(self):
        self.routing_system: Optional[DynamicRoutingSystem] = None
        self.evaluator: Optional[SystemEvaluator] = None
        self.query_manager = TestQueryManager()
        self.is_running = True

    def initialize_system(self):
        """Initialize the routing system"""
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
        """Run in interactive query mode"""
        print("\n🔄 Interactive Query Mode")
        print("Enter queries to test the routing system (type 'exit' to quit)")
        print("Commands: 'stats', 'cache', 'help', 'exit'")
        print("-" * 50)

        while self.is_running:
            try:
                user_input = input("\nQuery> ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                elif user_input.lower() == 'stats':
                    self._show_stats()
                elif user_input.lower() == 'cache':
                    self._show_cache_info()
                elif user_input.lower() == 'clear':
                    self._clear_screen()
                else:
                    # Process the query
                    self._process_interactive_query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _process_interactive_query(self, query: str):
        """Process a single interactive query"""
        print(f"\n📝 Processing: '{query[:60]}{'...' if len(query) > 60 else ''}'")

        start_time = time.time()
        result = self.routing_system.process_query(query)
        end_time = time.time()

        if result['success']:
            routing_info = result['routing_info']

            print(f"✅ Success!")
            print(f"🤖 Model Used: {routing_info['actual_model_used']}")
            print(f"💾 Cache Hit: {'Yes' if routing_info['cache_hit'] else 'No'}")
            print(f"🔄 Fallback: {'Yes' if routing_info['fallback_triggered'] else 'No'}")
            print(f"⏱️  Processing Time: {routing_info['processing_time']:.3f}s")
            print(f"💰 Resource Cost: {routing_info['resource_cost']}")
            print(f"📊 Total Time: {end_time - start_time:.3f}s")

            # Show classification info if available
            if 'classification' in routing_info and routing_info['classification']:
                class_info = routing_info['classification']
                print(f"🎯 Recommended: {class_info.get('recommended_model', 'N/A')}")
                print(f"🎲 Confidence: {class_info.get('confidence', 0):.2f}")

            print(f"\n📄 Response:")
            print(f"{result['response']}")

        else:
            print(f"❌ Failed: {result['error']}")

    def _show_help(self):
        """Show help information"""
        help_text = """
🔧 Available Commands:
  - Type any query to process it through the routing system
  - 'stats'  : Show system performance statistics  
  - 'cache'  : Show cache information
  - 'clear'  : Clear the screen
  - 'help'   : Show this help message
  - 'exit'   : Exit the program
  
🎯 Example Queries:
  Simple  : "What is Python?"
  Medium  : "How does machine learning work?"
  Advanced: "Design a distributed system architecture"
        """
        print(help_text)

    def _show_stats(self):
        """Show system statistics"""
        if not self.routing_system:
            print("❌ System not initialized")
            return

        print("\n📊 System Statistics:")
        stats = self.routing_system.get_detailed_stats()

        # System stats
        sys_stats = stats['system']
        print(f"  📈 Total Queries: {sys_stats['total_queries']}")
        print(f"  ✅ Successful: {sys_stats['successful_queries']}")
        print(f"  ❌ Failed: {sys_stats['failed_queries']}")
        print(f"  📊 Success Rate: {sys_stats['successful_queries']/max(sys_stats['total_queries'], 1):.1%}")

        # Cache stats
        cache_stats = stats['cache']
        print(f"  💾 Cache Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
        print(f"  🗄️  Cache Size: {cache_stats.get('current_cache_size', 0)}")

        # Model usage
        print(f"  🤖 Model Usage:")
        for model, count in sys_stats['model_usage'].items():
            percentage = count / max(sys_stats['total_queries'], 1) * 100
            print(f"     {model}: {count} ({percentage:.1f}%)")

    def _show_cache_info(self):
        """Show cache information"""
        if not self.routing_system:
            print("❌ System not initialized")
            return

        cache_info = self.routing_system.cache_manager.get_cache_info()
        print(f"\n💾 Cache Information:")
        print(f"  📊 Total Entries: {cache_info['total_entries']}")
        print(f"  ⚙️  Max Size: {cache_info['config']['max_size']}")
        print(f"  ⏰ TTL: {cache_info['config']['ttl_seconds']}s")

        if cache_info['entries']:
            print(f"  🔝 Most Accessed Entries:")
            for i, entry in enumerate(cache_info['entries'][:3], 1):
                print(f"     {i}. {entry['query']} (accessed {entry['access_count']} times)")

    def _clear_screen(self):
        """Clear the screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

    def run_evaluation(self, test_suite: str = 'basic', save_results: bool = True):
        """Run system evaluation"""
        print(f"\n🔍 Running {test_suite.upper()} evaluation...")

        # Get test queries
        if test_suite == 'quick':
            queries = quick_test_queries()
        elif test_suite == 'comprehensive':
            queries = comprehensive_test_queries()
        elif test_suite in COMPREHENSIVE_TEST_SUITE:
            queries = self.query_manager.get_query_collection(test_suite)
        else:
            print(f"❌ Unknown test suite: {test_suite}")
            return

        print(f"📝 Testing with {len(queries)} queries...")

        # Run evaluation
        start_time = time.time()
        results = self.evaluator.evaluate_system(queries)
        end_time = time.time()

        # Show results
        print(f"\n✅ Evaluation completed in {end_time - start_time:.2f} seconds")
        self._display_evaluation_results(results)

        return results

    def _display_evaluation_results(self, results: Dict[str, Any]):
        """Display evaluation results"""
        summary = results.get('evaluation_summary', {})

        print(f"\n📊 EVALUATION RESULTS:")
        print(f"  📈 Total Queries: {summary.get('total_queries_evaluated', 0)}")
        print(f"  ✅ Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"  💾 Cache Hit Rate: {summary.get('cache_hit_rate', 0):.1%}")
        print(f"  🔄 Fallback Rate: {summary.get('fallback_rate', 0):.1%}")
        print(f"  ⏱️  Avg Processing Time: {summary.get('average_processing_time', 0):.3f}s")
        print(f"  💰 Avg Resource Cost: {summary.get('average_resource_cost', 0):.1f}")

        # Show recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

    def run_comparison_demo(self):
        """Run a demonstration comparing routing vs direct model usage"""
        print("\n🔀 Running Routing vs Direct Model Comparison...")

        demo_queries = [
            "What is Python?",
            "How does machine learning work?",
            "Design a distributed system architecture for scalability"
        ]

        print(f"Testing {len(demo_queries)} queries...")

        for i, query in enumerate(demo_queries, 1):
            print(f"\n--- Query {i}: {query} ---")

            # Test with routing system
            routing_result = self.routing_system.process_query(query)
            routing_time = routing_result['routing_info']['processing_time']
            routing_cost = routing_result['routing_info']['resource_cost']
            routing_model = routing_result['routing_info']['actual_model_used']

            print(f"🎯 Routing System:")
            print(f"   Model: {routing_model}")
            print(f"   Time: {routing_time:.3f}s")
            print(f"   Cost: {routing_cost}")

            # Test with direct advanced model (comparison)
            all_models = ModelFactory.create_all_models()
            direct_result = all_models['advanced'].process_query(query)

            print(f"🔧 Always Advanced:")
            print(f"   Model: advanced")
            print(f"   Time: {direct_result['response_time']:.3f}s")
            print(f"   Cost: {direct_result['resource_cost']}")

            # Calculate savings
            time_savings = direct_result['response_time'] - routing_time
            cost_savings = direct_result['resource_cost'] - routing_cost

            print(f"💡 Savings:")
            print(f"   Time: {time_savings:.3f}s ({time_savings/direct_result['response_time']*100:.1f}%)")
            print(f"   Cost: {cost_savings} ({cost_savings/direct_result['resource_cost']*100:.1f}%)")

    def shutdown(self):
        """Shutdown the system gracefully"""
        print("\n🔄 Shutting down system...")
        if self.routing_system:
            self.routing_system.shutdown()
        print("✅ Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Dynamic Routing System for LLM Queries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive                    # Interactive mode
  python main.py --evaluate basic                 # Basic evaluation
  python main.py --evaluate comprehensive         # Full evaluation  
  python main.py --query "What is Python?"        # Single query
  python main.py --demo                           # Comparison demo
        """
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--evaluate', '-e',
        choices=['quick', 'basic', 'comprehensive', 'edge_cases', 'domain_specific', 'all'],
        help='Run system evaluation with specified test suite'
    )

    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Process a single query'
    )

    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run comparison demonstration'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show available models and configuration'
    )

    args = parser.parse_args()

    # Show banner
    print("=" * 60)
    print("🤖 DYNAMIC ROUTING SYSTEM FOR LLM QUERIES")
    print("    Intelligent Query Routing with Caching & Fallback")
    print("=" * 60)

    # Initialize CLI
    cli = DynamicRoutingCLI()

    try:
        # Initialize system
        if not cli.initialize_system():
            sys.exit(1)

        # Handle arguments
        if args.stats:
            print(f"\n📋 Available Models: {', '.join(get_all_model_names())}")
            print(f"🔧 Debug Mode: {GENERAL_SETTINGS.get('debug_mode', False)}")
            print(f"🗣️  Verbose Logging: {GENERAL_SETTINGS.get('verbose_logging', True)}")

        elif args.query:
            # Single query mode
            print(f"\n🔍 Processing single query...")
            result = cli.routing_system.process_query(args.query)

            if result['success']:
                print(f"✅ Result: {result['response']}")
                print(f"🤖 Model: {result['routing_info']['actual_model_used']}")
                print(f"⏱️  Time: {result['routing_info']['processing_time']:.3f}s")
            else:
                print(f"❌ Error: {result['error']}")

        elif args.evaluate:
            # Evaluation mode
            cli.run_evaluation(args.evaluate)

        elif args.demo:
            # Demonstration mode
            cli.run_comparison_demo()

        elif args.interactive:
            # Interactive mode
            cli.run_interactive_mode()

        else:
            # Default: show quick demo
            print("\n🚀 Running Quick Demo...")
            print("Use --help for more options")

            # Show quick stats
            print(f"\n📋 System Ready!")
            print(f"   Available Models: {', '.join(get_all_model_names())}")

            # Run quick evaluation
            print("\n🔍 Quick Evaluation...")
            cli.run_evaluation('quick')

            print("\n💡 Try: python main.py --interactive")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if GENERAL_SETTINGS.get('debug_mode', False):
            import traceback
            traceback.print_exc()

    finally:
        cli.shutdown()


if __name__ == "__main__":
    main()