"""
Test Query Collections for Dynamic Routing System
Contains diverse query sets for testing and evaluation purposes
"""

import json
import random
from typing import List, Dict, Any
from config import FILE_PATHS


class TestQueryGenerator:
    """Generates test queries for system evaluation"""

    def __init__(self):
        self.query_templates = {
            'simple': [
                "What is {topic}?",
                "Define {topic}",
                "Who invented {topic}?",
                "When was {topic} created?",
                "Where is {topic} used?",
                "Is {topic} important?",
                "How much does {topic} cost?",
                "What does {topic} mean?",
                "Can you explain {topic}?",
                "Tell me about {topic}"
            ],
            'medium': [
                "How does {topic} work?",
                "Explain the process of {topic}",
                "What are the advantages and disadvantages of {topic}?",
                "Compare {topic1} and {topic2}",
                "What are the steps to implement {topic}?",
                "How can I learn {topic} effectively?",
                "What are the main features of {topic}?",
                "Describe the relationship between {topic1} and {topic2}",
                "What are some examples of {topic} in practice?",
                "How has {topic} evolved over time?"
            ],
            'advanced': [
                "Analyze the computational complexity of {topic} and propose optimization strategies",
                "Design a comprehensive architecture for {topic} considering scalability and performance",
                "Evaluate the theoretical foundations of {topic} and discuss potential improvements",
                "Develop a detailed algorithm for {topic} with complexity analysis and edge case handling",
                "Research and compare multiple approaches to {topic} with quantitative analysis",
                "Create a mathematical model for {topic} and derive optimal solutions",
                "Investigate the security implications of {topic} and propose mitigation strategies",
                "Design and implement a distributed system for {topic} with fault tolerance",
                "Conduct a comprehensive literature review on {topic} and identify research gaps",
                "Optimize the performance of {topic} using advanced techniques and provide benchmarks"
            ]
        }

        self.topics = {
            'programming': [
                'binary search', 'machine learning', 'neural networks', 'algorithms',
                'data structures', 'databases', 'web development', 'Python',
                'JavaScript', 'API design', 'microservices', 'cloud computing'
            ],
            'mathematics': [
                'calculus', 'linear algebra', 'statistics', 'probability',
                'discrete math', 'optimization', 'game theory', 'cryptography'
            ],
            'science': [
                'quantum computing', 'artificial intelligence', 'robotics',
                'biotechnology', 'nanotechnology', 'renewable energy'
            ],
            'general': [
                'photography', 'cooking', 'music', 'history', 'geography',
                'literature', 'philosophy', 'psychology', 'economics'
            ]
        }

    def generate_queries(self, count_per_complexity: int = 5) -> List[Dict[str, Any]]:
        """Generate random test queries"""
        queries = []

        for complexity in ['simple', 'medium', 'advanced']:
            for _ in range(count_per_complexity):
                template = random.choice(self.query_templates[complexity])

                # Get random topics
                all_topics = []
                for topic_list in self.topics.values():
                    all_topics.extend(topic_list)

                topic = random.choice(all_topics)

                # Handle templates with multiple topics
                if '{topic1}' in template and '{topic2}' in template:
                    topic1 = random.choice(all_topics)
                    topic2 = random.choice(all_topics)
                    while topic2 == topic1:  # Ensure different topics
                        topic2 = random.choice(all_topics)
                    query = template.format(topic1=topic1, topic2=topic2)
                else:
                    query = template.format(topic=topic)

                queries.append({
                    'query': query,
                    'expected_complexity': complexity,
                    'domain': self._get_topic_domain(topic),
                    'generated': True
                })

        return queries

    def _get_topic_domain(self, topic: str) -> str:
        """Get the domain of a topic"""
        for domain, topics in self.topics.items():
            if topic in topics:
                return domain
        return 'general'


# Pre-defined test query collections
BASIC_TEST_QUERIES = [
    {
        'query': 'What is Python?',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Basic factual question about a programming language'
    },
    {
        'query': 'How do I create a list in Python?',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Simple how-to question'
    },
    {
        'query': 'What is machine learning?',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Basic definition question'
    },
    {
        'query': 'Explain how neural networks work',
        'expected_complexity': 'medium',
        'domain': 'programming',
        'description': 'Medium complexity explanation request'
    },
    {
        'query': 'How do I implement a binary search algorithm in Python?',
        'expected_complexity': 'medium',
        'domain': 'programming',
        'description': 'Implementation-focused medium complexity query'
    },
    {
        'query': 'Compare supervised and unsupervised learning approaches',
        'expected_complexity': 'medium',
        'domain': 'programming',
        'description': 'Comparison and analysis request'
    },
    {
        'query': 'Analyze the computational complexity of different sorting algorithms and design an optimal solution for large datasets',
        'expected_complexity': 'advanced',
        'domain': 'programming',
        'description': 'Complex analysis and design task'
    },
    {
        'query': 'Design a distributed system architecture for real-time data processing with fault tolerance and scalability considerations',
        'expected_complexity': 'advanced',
        'domain': 'programming',
        'description': 'Advanced system design task'
    },
    {
        'query': 'Develop a mathematical model for optimizing resource allocation in cloud computing environments',
        'expected_complexity': 'advanced',
        'domain': 'programming',
        'description': 'Advanced mathematical modeling task'
    }
]

EDGE_CASE_QUERIES = [
    {
        'query': '',
        'expected_complexity': 'simple',
        'domain': 'general',
        'description': 'Empty query test'
    },
    {
        'query': '   ',
        'expected_complexity': 'simple',
        'domain': 'general',
        'description': 'Whitespace-only query'
    },
    {
        'query': 'a',
        'expected_complexity': 'simple',
        'domain': 'general',
        'description': 'Single character query'
    },
    {
        'query': '?' * 100,
        'expected_complexity': 'simple',
        'domain': 'general',
        'description': 'Repetitive character query'
    },
    {
        'query': 'This is a very long query that contains a lot of text but doesn\'t really ask anything specific or meaningful and is just designed to test how the system handles verbose but ultimately simple requests that go on and on without much substance or clear intent behind them.',
        'expected_complexity': 'simple',
        'domain': 'general',
        'description': 'Very long but simple query'
    },
    {
        'query': 'Write code debug fix error solve problem algorithm optimize performance analyze data implement solution design architecture',
        'expected_complexity': 'advanced',
        'domain': 'programming',
        'description': 'Keyword-heavy query'
    }
]

DOMAIN_SPECIFIC_QUERIES = [
    # Programming domain
    {
        'query': 'What is a variable in Python?',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Basic programming concept'
    },
    {
        'query': 'How to implement exception handling in Python?',
        'expected_complexity': 'medium',
        'domain': 'programming',
        'description': 'Programming implementation question'
    },
    {
        'query': 'Design a high-performance concurrent web crawler with rate limiting and distributed processing capabilities',
        'expected_complexity': 'advanced',
        'domain': 'programming',
        'description': 'Advanced programming system design'
    },

    # Mathematics domain
    {
        'query': 'What is calculus?',
        'expected_complexity': 'simple',
        'domain': 'mathematics',
        'description': 'Basic math definition'
    },
    {
        'query': 'Explain the fundamental theorem of calculus',
        'expected_complexity': 'medium',
        'domain': 'mathematics',
        'description': 'Mathematical theorem explanation'
    },
    {
        'query': 'Derive and prove the convergence conditions for the Newton-Raphson method in multidimensional optimization',
        'expected_complexity': 'advanced',
        'domain': 'mathematics',
        'description': 'Advanced mathematical derivation'
    },

    # Science domain
    {
        'query': 'What is DNA?',
        'expected_complexity': 'simple',
        'domain': 'science',
        'description': 'Basic science definition'
    },
    {
        'query': 'How does photosynthesis work?',
        'expected_complexity': 'medium',
        'domain': 'science',
        'description': 'Scientific process explanation'
    },
    {
        'query': 'Analyze the quantum mechanical principles underlying superconductivity and design novel materials for room-temperature applications',
        'expected_complexity': 'advanced',
        'domain': 'science',
        'description': 'Advanced scientific analysis and design'
    },

    # General domain
    {
        'query': 'What is cooking?',
        'expected_complexity': 'simple',
        'domain': 'general',
        'description': 'Basic general knowledge'
    },
    {
        'query': 'How do I bake a cake?',
        'expected_complexity': 'medium',
        'domain': 'general',
        'description': 'General how-to question'
    },
    {
        'query': 'Develop a comprehensive culinary curriculum that integrates molecular gastronomy techniques with traditional cooking methods',
        'expected_complexity': 'advanced',
        'domain': 'general',
        'description': 'Advanced general knowledge application'
    }
]

CACHING_TEST_QUERIES = [
    {
        'query': 'What is Python programming?',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Cache test query 1'
    },
    {
        'query': 'What is Python programming language?',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Similar to cache test query 1 - should trigger similarity matching'
    },
    {
        'query': 'Python programming language definition',
        'expected_complexity': 'simple',
        'domain': 'programming',
        'description': 'Another similar query for cache testing'
    },
    {
        'query': 'How to implement sorting algorithms?',
        'expected_complexity': 'medium',
        'domain': 'programming',
        'description': 'Cache test query 2'
    },
    {
        'query': 'Implementation of sorting algorithms',
        'expected_complexity': 'medium',
        'domain': 'programming',
        'description': 'Similar to cache test query 2'
    }
]

FALLBACK_TEST_QUERIES = [
    {
        'query': 'Create a complex distributed system with microservices architecture, implement advanced security protocols, optimize for high availability and fault tolerance, design comprehensive monitoring and logging systems, and ensure scalability for millions of concurrent users while maintaining sub-millisecond response times',
        'expected_complexity': 'advanced',
        'domain': 'programming',
        'description': 'Very complex query likely to trigger fallback mechanisms'
    },
    {
        'query': 'Analyze quantum computing algorithms for cryptographic applications',
        'expected_complexity': 'advanced',
        'domain': 'science',
        'description': 'Complex interdisciplinary query'
    }
]

# Combined test suites
ALL_TEST_QUERIES = (
        BASIC_TEST_QUERIES +
        EDGE_CASE_QUERIES +
        DOMAIN_SPECIFIC_QUERIES +
        CACHING_TEST_QUERIES +
        FALLBACK_TEST_QUERIES
)

COMPREHENSIVE_TEST_SUITE = {
    'basic': BASIC_TEST_QUERIES,
    'edge_cases': EDGE_CASE_QUERIES,
    'domain_specific': DOMAIN_SPECIFIC_QUERIES,
    'caching_tests': CACHING_TEST_QUERIES,
    'fallback_tests': FALLBACK_TEST_QUERIES,
    'all': ALL_TEST_QUERIES
}


class TestQueryManager:
    """Manages test query collections and provides utilities"""

    def __init__(self):
        self.generator = TestQueryGenerator()
        self.query_collections = COMPREHENSIVE_TEST_SUITE

    def get_query_collection(self, collection_name: str = 'all') -> List[Dict[str, Any]]:
        """Get a specific query collection"""
        return self.query_collections.get(collection_name, self.query_collections['all'])

    def get_random_queries(self, count: int = 10, complexity: str = None) -> List[Dict[str, Any]]:
        """Get random queries from all collections"""
        all_queries = self.query_collections['all']

        if complexity:
            filtered_queries = [q for q in all_queries if q['expected_complexity'] == complexity]
            all_queries = filtered_queries if filtered_queries else all_queries

        if count >= len(all_queries):
            return all_queries

        return random.sample(all_queries, count)

    def generate_custom_queries(self, count_per_complexity: int = 5) -> List[Dict[str, Any]]:
        """Generate custom queries using templates"""
        return self.generator.generate_queries(count_per_complexity)

    def get_queries_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get queries filtered by domain"""
        return [q for q in self.query_collections['all'] if q.get('domain') == domain]

    def get_queries_by_complexity(self, complexity: str) -> List[Dict[str, Any]]:
        """Get queries filtered by complexity"""
        return [q for q in self.query_collections['all'] if q['expected_complexity'] == complexity]

    def save_queries_to_file(self, queries: List[Dict[str, Any]], filename: str = None):
        """Save query collection to JSON file"""
        if filename is None:
            filename = FILE_PATHS.get('test_queries_file', 'test_queries.json')

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(queries, f, indent=2, ensure_ascii=False)
            print(f"Queries saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving queries: {e}")
            return False

    def load_queries_from_file(self, filename: str = None) -> List[Dict[str, Any]]:
        """Load query collection from JSON file"""
        if filename is None:
            filename = FILE_PATHS.get('test_queries_file', 'test_queries.json')

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            print(f"Queries loaded from {filename}")
            return queries
        except FileNotFoundError:
            print(f"File {filename} not found")
            return []
        except Exception as e:
            print(f"Error loading queries: {e}")
            return []

    def analyze_query_distribution(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of queries"""
        if not queries:
            return {'error': 'No queries provided'}

        complexity_counts = {}
        domain_counts = {}
        length_stats = []

        for query in queries:
            # Complexity distribution
            complexity = query.get('expected_complexity', 'unknown')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

            # Domain distribution
            domain = query.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

            # Length statistics
            length_stats.append(len(query.get('query', '')))

        return {
            'total_queries': len(queries),
            'complexity_distribution': complexity_counts,
            'domain_distribution': domain_counts,
            'length_statistics': {
                'min_length': min(length_stats) if length_stats else 0,
                'max_length': max(length_stats) if length_stats else 0,
                'avg_length': sum(length_stats) / len(length_stats) if length_stats else 0,
                'median_length': sorted(length_stats)[len(length_stats)//2] if length_stats else 0
            }
        }

    def create_balanced_test_set(self, total_queries: int = 30) -> List[Dict[str, Any]]:
        """Create a balanced test set with equal representation"""
        queries_per_complexity = total_queries // 3
        remaining = total_queries % 3

        balanced_queries = []

        for complexity in ['simple', 'medium', 'advanced']:
            complexity_queries = self.get_queries_by_complexity(complexity)

            # Add extra query to advanced if there's a remainder
            count = queries_per_complexity + (1 if complexity == 'advanced' and remaining > 0 else 0)

            if len(complexity_queries) >= count:
                selected = random.sample(complexity_queries, count)
            else:
                # If not enough queries, use all available and generate more
                selected = complexity_queries.copy()
                needed = count - len(selected)
                generated = self.generator.generate_queries(needed)
                generated_complexity = [q for q in generated if q['expected_complexity'] == complexity]
                selected.extend(generated_complexity[:needed])

            balanced_queries.extend(selected)

        return balanced_queries


def quick_test_queries() -> List[Dict[str, Any]]:
    """Quick function to get a small set of test queries"""
    return BASIC_TEST_QUERIES[:5]


def comprehensive_test_queries() -> List[Dict[str, Any]]:
    """Get comprehensive test query set"""
    manager = TestQueryManager()
    return manager.create_balanced_test_set(30)


if __name__ == "__main__":
    print("Testing Query Collections...")

    # Initialize manager
    manager = TestQueryManager()

    # Test different collections
    print("\n=== Query Collection Analysis ===")
    for name, collection in COMPREHENSIVE_TEST_SUITE.items():
        analysis = manager.analyze_query_distribution(collection)
        print(f"\n{name.upper()} Collection:")
        print(f"  Total queries: {analysis['total_queries']}")
        print(f"  Complexity distribution: {analysis['complexity_distribution']}")
        print(f"  Domain distribution: {analysis['domain_distribution']}")
        print(f"  Average length: {analysis['length_statistics']['avg_length']:.1f} chars")

    # Test query generation
    print("\n=== Generated Queries Sample ===")
    generated = manager.generate_custom_queries(2)
    for i, query in enumerate(generated[:6], 1):
        print(f"{i}. [{query['expected_complexity']}] {query['query']}")

    # Test balanced set creation
    print("\n=== Balanced Test Set ===")
    balanced = manager.create_balanced_test_set(12)
    analysis = manager.analyze_query_distribution(balanced)
    print(f"Complexity distribution: {analysis['complexity_distribution']}")

    # Save sample queries
    sample_file = "sample_test_queries.json"
    manager.save_queries_to_file(balanced, sample_file)
    print(f"\nSample queries saved to {sample_file}")

    print("\nTest query collections ready!")