# test_queries.py

from typing import List, Dict, Any

BASIC_TEST_QUERIES: List[Dict[str, Any]] = [
    {'query': 'What is Python?', 'expected_complexity': 'simple'},
    {'query': 'How do I create a list in Python?', 'expected_complexity': 'medium'},
    {'query': 'What is machine learning?', 'expected_complexity': 'simple'},
    {'query': 'Explain how neural networks work', 'expected_complexity': 'medium'},
    {'query': 'Analyze the computational complexity of different sorting algorithms', 'expected_complexity': 'advanced'},
    {'query': 'Who invented the light bulb?', 'expected_complexity': 'simple'},
    {'query': 'Compare SQL and NoSQL databases', 'expected_complexity': 'medium'}
]

EDGE_CASE_QUERIES: List[Dict[str, Any]] = [
    {'query': '', 'expected_complexity': 'simple'},
    {'query': '   ', 'expected_complexity': 'simple'},
    {'query': 'Test', 'expected_complexity': 'simple'},
    {'query': 'analyze evaluate design optimize'*10, 'expected_complexity': 'advanced'},
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
    """Returns a small subset of queries for a fast evaluation."""
    return BASIC_TEST_QUERIES[:5]

def comprehensive_test_queries() -> List[Dict[str, Any]]:
    """Returns a larger, more comprehensive set of queries."""
    return BASIC_TEST_QUERIES