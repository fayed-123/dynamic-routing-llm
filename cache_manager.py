"""
Cache Manager for Dynamic Routing System
Handles caching of query results to improve performance and reduce resource usage
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from config import get_cache_config, FILE_PATHS


class CacheEntry:
    """Represents a single cache entry with metadata"""

    def __init__(self, query: str, result: Dict[Any, Any], model_used: str):
        self.query = query
        self.result = result
        self.model_used = model_used
        self.timestamp = time.time()
        self.access_count = 1
        self.last_accessed = self.timestamp
        self.query_hash = self._generate_hash(query)

    def _generate_hash(self, query: str) -> str:
        """Generate a unique hash for the query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if the cache entry has expired"""
        return (time.time() - self.timestamp) > ttl_seconds

    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization"""
        return {
            'query': self.query,
            'result': self.result,
            'model_used': self.model_used,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'query_hash': self.query_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create cache entry from dictionary"""
        entry = cls.__new__(cls)
        entry.query = data['query']
        entry.result = data['result']
        entry.model_used = data['model_used']
        entry.timestamp = data['timestamp']
        entry.access_count = data['access_count']
        entry.last_accessed = data['last_accessed']
        entry.query_hash = data['query_hash']
        return entry


class CacheManager:
    """Manages caching of query results with LRU eviction and similarity matching"""

    def __init__(self):
        self.config = get_cache_config()
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []  # For LRU tracking

        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_saves': 0,
            'evictions': 0,
            'similarity_matches': 0
        }

        # Load existing cache if available
        self._load_cache_from_file()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query
        Returns None if not found or expired
        """
        self.stats['total_requests'] += 1

        if not self.config['enabled']:
            self.stats['cache_misses'] += 1
            return None

        query_normalized = query.lower().strip()
        query_hash = hashlib.md5(query_normalized.encode()).hexdigest()

        # Direct hash match
        if query_hash in self.cache:
            entry = self.cache[query_hash]

            # Check if expired
            if entry.is_expired(self.config['ttl_seconds']):
                self._remove_entry(query_hash)
                self.stats['cache_misses'] += 1
                return None

            # Update access and return result
            entry.update_access()
            self._update_access_order(query_hash)
            self.stats['cache_hits'] += 1

            return {
                'result': entry.result,
                'model_used': entry.model_used,
                'cached_at': entry.timestamp,
                'access_count': entry.access_count,
                'cache_hit': True
            }

        # Try similarity matching
        similar_result = self._find_similar_query(query_normalized)
        if similar_result:
            self.stats['cache_hits'] += 1
            self.stats['similarity_matches'] += 1
            return similar_result

        self.stats['cache_misses'] += 1
        return None

    def put(self, query: str, result: Dict[str, Any], model_used: str) -> bool:
        """
        Cache a query result
        Returns True if successfully cached
        """
        if not self.config['enabled']:
            return False

        try:
            query_normalized = query.lower().strip()
            query_hash = hashlib.md5(query_normalized.encode()).hexdigest()

            # Check if we need to evict entries
            if len(self.cache) >= self.config['max_size']:
                self._evict_lru_entries()

            # Create and store cache entry
            entry = CacheEntry(query, result, model_used)
            self.cache[query_hash] = entry
            self._update_access_order(query_hash)

            self.stats['cache_saves'] += 1
            return True

        except Exception as e:
            print(f"Error caching query result: {e}")
            return False

    def _find_similar_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Find and return result from a similar cached query"""
        best_match = None
        best_similarity = 0.0
        threshold = self.config['similarity_threshold']

        for entry in self.cache.values():
            if entry.is_expired(self.config['ttl_seconds']):
                continue

            similarity = self._calculate_similarity(query, entry.query.lower().strip())

            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match:
            best_match.update_access()
            self._update_access_order(best_match.query_hash)

            return {
                'result': best_match.result,
                'model_used': best_match.model_used,
                'cached_at': best_match.timestamp,
                'access_count': best_match.access_count,
                'cache_hit': True,
                'similarity_match': True,
                'similarity_score': best_similarity,
                'original_query': best_match.query
            }

        return None

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two queries using word overlap
        Returns similarity score between 0.0 and 1.0
        """
        # Simple word-based similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard_similarity = len(intersection) / len(union)

        # Length similarity bonus (prefer similar-length queries)
        len_diff = abs(len(query1) - len(query2))
        max_len = max(len(query1), len(query2))
        length_similarity = 1.0 - (len_diff / max(max_len, 1))

        # Combined similarity score
        combined_similarity = (jaccard_similarity * 0.8) + (length_similarity * 0.2)

        return combined_similarity

    def _evict_lru_entries(self):
        """Evict least recently used entries to make space"""
        entries_to_remove = max(1, len(self.cache) // 4)  # Remove 25% when full

        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )

        for i in range(min(entries_to_remove, len(sorted_entries))):
            query_hash = sorted_entries[i][0]
            self._remove_entry(query_hash)
            self.stats['evictions'] += 1

    def _remove_entry(self, query_hash: str):
        """Remove an entry from cache"""
        if query_hash in self.cache:
            del self.cache[query_hash]
        if query_hash in self.access_order:
            self.access_order.remove(query_hash)

    def _update_access_order(self, query_hash: str):
        """Update LRU access order"""
        if query_hash in self.access_order:
            self.access_order.remove(query_hash)
        self.access_order.append(query_hash)

    def clear_cache(self):
        """Clear all cached entries"""
        cleared_count = len(self.cache)
        self.cache.clear()
        self.access_order.clear()
        return cleared_count

    def clear_expired(self) -> int:
        """Remove expired entries and return count of removed entries"""
        current_time = time.time()
        expired_hashes = []

        for query_hash, entry in self.cache.items():
            if entry.is_expired(self.config['ttl_seconds']):
                expired_hashes.append(query_hash)

        for query_hash in expired_hashes:
            self._remove_entry(query_hash)

        return len(expired_hashes)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = max(self.stats['total_requests'], 1)

        return {
            'total_requests': self.stats['total_requests'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / total_requests,
            'miss_rate': self.stats['cache_misses'] / total_requests,
            'cache_saves': self.stats['cache_saves'],
            'evictions': self.stats['evictions'],
            'similarity_matches': self.stats['similarity_matches'],
            'current_cache_size': len(self.cache),
            'max_cache_size': self.config['max_size'],
            'cache_usage': len(self.cache) / self.config['max_size']
        }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        entries_info = []

        for entry in self.cache.values():
            entries_info.append({
                'query': entry.query[:50] + '...' if len(entry.query) > 50 else entry.query,
                'model_used': entry.model_used,
                'access_count': entry.access_count,
                'age_seconds': time.time() - entry.timestamp,
                'last_accessed_seconds_ago': time.time() - entry.last_accessed
            })

        # Sort by access count (most accessed first)
        entries_info.sort(key=lambda x: x['access_count'], reverse=True)

        return {
            'total_entries': len(self.cache),
            'config': self.config,
            'entries': entries_info[:10]  # Top 10 most accessed
        }

    def _load_cache_from_file(self):
        """Load cache from persistent storage"""
        try:
            cache_file = FILE_PATHS['cache_file']
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for entry_data in data.get('entries', []):
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired(self.config['ttl_seconds']):
                    self.cache[entry.query_hash] = entry
                    self.access_order.append(entry.query_hash)

        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # No existing cache file or corrupted file
            pass

    def save_cache_to_file(self):
        """Save cache to persistent storage"""
        try:
            cache_file = FILE_PATHS['cache_file']
            data = {
                'timestamp': time.time(),
                'config': self.config,
                'entries': [entry.to_dict() for entry in self.cache.values()]
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error saving cache to file: {e}")
            return False

    def __del__(self):
        """Save cache when object is destroyed"""
        if hasattr(self, 'cache') and self.cache:
            self.save_cache_to_file()


# Utility functions
def create_cache_manager() -> CacheManager:
    """Factory function to create a cache manager instance"""
    return CacheManager()


def test_cache_similarity():
    """Test function for cache similarity matching"""
    cache = CacheManager()

    # Test queries
    test_cases = [
        ("What is Python?", "What is Python programming?"),
        ("How to sort arrays", "How to sort an array"),
        ("Machine learning algorithms", "ML algorithms"),
        ("Simple question", "Complex analysis question")
    ]

    print("Testing cache similarity:")
    for query1, query2 in test_cases:
        similarity = cache._calculate_similarity(query1, query2)
        print(f"'{query1}' vs '{query2}': {similarity:.3f}")


if __name__ == "__main__":
    print("Testing Cache Manager...")
    test_cache_similarity()