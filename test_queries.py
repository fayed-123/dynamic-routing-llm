# test_queries.py (النسخة النهائية والمصلحة)

from typing import List, Dict, Any

# A larger and more diverse list of queries to improve model accuracy.
BASIC_TEST_QUERIES: List[Dict[str, Any]] = [
    # --- Simple Queries (15 examples) ---
    {'query': 'What is Python?', 'expected_complexity': 'simple'},
    {'query': 'Who is the CEO of Tesla?', 'expected_complexity': 'simple'},
    {'query': 'Define object-oriented programming', 'expected_complexity': 'simple'},
    {'query': 'What is the capital of France?', 'expected_complexity': 'simple'},
    {'query': 'List the planets in the solar system', 'expected_complexity': 'simple'},
    {'query': 'When was the internet invented?', 'expected_complexity': 'simple'},
    {'query': 'What does CPU stand for?', 'expected_complexity': 'simple'},
    {'query': 'Who wrote "Hamlet"?', 'expected_complexity': 'simple'},
    {'query': 'Define the word "heuristic"', 'expected_complexity': 'simple'},
    {'query': 'What is the formula for water?', 'expected_complexity': 'simple'},
    {'query': 'Where is the Great Wall of China?', 'expected_complexity': 'simple'},
    {'query': 'What color is the sky?', 'expected_complexity': 'simple'},
    {'query': 'Who was Albert Einstein?', 'expected_complexity': 'simple'},
    {'query': 'Define "API"', 'expected_complexity': 'simple'},
    {'query': 'What is the largest ocean?', 'expected_complexity': 'simple'},

    # --- Medium Queries (15 examples) ---
    {'query': 'How do I create a list in Python?', 'expected_complexity': 'medium'},
    {'query': 'Explain how neural networks work', 'expected_complexity': 'medium'},
    {'query': 'Compare SQL and NoSQL databases', 'expected_complexity': 'medium'},
    {'query': 'What are the steps to build a simple web server?', 'expected_complexity': 'medium'},
    {'query': 'Why is the sky blue?', 'expected_complexity': 'medium'},
    {'query': 'Describe the process of photosynthesis', 'expected_complexity': 'medium'},
    {'query': 'How to implement a sorting algorithm in Java?', 'expected_complexity': 'medium'},
    {'query': 'Explain the difference between a GET and POST request', 'expected_complexity': 'medium'},
    {'query': 'Summarize the plot of the movie Inception', 'expected_complexity': 'medium'},
    {'query': 'How does a blockchain work?', 'expected_complexity': 'medium'},
    {'query': 'Explain the concept of supply and demand', 'expected_complexity': 'medium'},
    {'query': 'How to change a car tire?', 'expected_complexity': 'medium'},
    {'query': 'Describe the main features of object-oriented programming', 'expected_complexity': 'medium'},
    {'query': 'Why should I use a virtual environment in Python?', 'expected_complexity': 'medium'},
    {'query': 'Compare the benefits of running code on a CPU vs a GPU', 'expected_complexity': 'medium'},

    # --- Advanced Queries (15 examples) ---
    {'query': 'Analyze the computational complexity of different sorting algorithms', 'expected_complexity': 'advanced'},
    {'query': 'Design a scalable microservices architecture for an e-commerce platform', 'expected_complexity': 'advanced'},
    {'query': 'Evaluate the impact of AI on the future of employment', 'expected_complexity': 'advanced'},
    {'query': 'Write a detailed algorithm for a traveling salesman problem', 'expected_complexity': 'advanced'},
    {'query': 'Develop a strategy for optimizing database performance under high load', 'expected_complexity': 'advanced'},
    {'query': 'Critique the design of the TCP/IP protocol suite', 'expected_complexity': 'advanced'},
    {'query': 'Analyze the ethical implications of genetic engineering', 'expected_complexity': 'advanced'},
    {'query': 'Design a fault-tolerant distributed caching system', 'expected_complexity': 'advanced'},
    {'query': 'Optimize the following Python code for performance and memory usage', 'expected_complexity': 'advanced'},
    {'query': 'Propose a machine learning model for stock price prediction and justify your choice', 'expected_complexity': 'advanced'},
    {'query': 'Design a database schema for a social media application with followers and posts', 'expected_complexity': 'advanced'},
    {'query': 'Evaluate the effectiveness of different cybersecurity protocols against DDoS attacks', 'expected_complexity': 'advanced'},
    {'query': 'Analyze the trade-offs between monolithic and microservices architectures', 'expected_complexity': 'advanced'},
    {'query': 'Develop a comprehensive plan for migrating a legacy system to a cloud-native architecture', 'expected_complexity': 'advanced'},
    {'query': 'Design an A/B testing framework to evaluate new features in a web application', 'expected_complexity': 'advanced'},
]

EDGE_CASE_QUERIES: List[Dict[str, Any]] = [
    {'query': '', 'expected_complexity': 'simple'},
    {'query': '   ', 'expected_complexity': 'simple'},
    {'query': 'Test', 'expected_complexity': 'simple'},
    {'query': 'analyze evaluate design optimize ' * 5, 'expected_complexity': 'advanced'},
]

ALL_QUERIES = BASIC_TEST_QUERIES + EDGE_CASE_QUERIES

COMPREHENSIVE_TEST_SUITE: Dict[str, List[Dict[str, Any]]] = {
    'basic': BASIC_TEST_QUERIES,
    'edge_cases': EDGE_CASE_QUERIES,
    'all': ALL_QUERIES
}

class TestQueryManager:
    """Manages collections of test queries."""
    def get_query_collection(self, collection_name: str = 'all') -> List[Dict[str, Any]]:
        """Returns a specific list of test queries."""
        return COMPREHENSIVE_TEST_SUITE.get(collection_name, [])

def quick_test_queries() -> List[Dict[str, Any]]:
    """
    Returns a small but balanced subset of queries for a fast, representative evaluation.
    """
    return [
        {'query': 'What is the capital of France?', 'expected_complexity': 'simple'},
        {'query': 'How does a blockchain work?', 'expected_complexity': 'medium'},
        {'query': 'Analyze the ethical implications of genetic engineering', 'expected_complexity': 'advanced'},
    ]

def comprehensive_test_queries() -> List[Dict[str, Any]]:
    """
    Returns a larger, more comprehensive set of ALL queries.
    --- THIS IS THE CORRECTED PART ---
    """
    return ALL_QUERIES