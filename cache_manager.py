
import json
import time
import hashlib
from typing import Dict, Any, Optional
from config import get_cache_config, FILE_PATHS

class CacheEntry:
    """Represents a single cache entry with metadata."""
    def __init__(self, query: str, result: Dict, model_used: str):
        self.query = query
        self.result = result
        self.model_used = model_used
        self.timestamp = time.time()
        self.access_count = 1
        self.last_accessed = self.timestamp
        self.query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.timestamp) > ttl_seconds

    def update_access(self):
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        entry = cls.__new__(cls)
        entry.__dict__.update(data)
        return entry

class CacheManager:
    """Manages caching with LRU eviction, TTL, and similarity matching."""
    def __init__(self):
        self.config = get_cache_config()
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            'total_requests': 0, 'cache_hits': 0, 'cache_misses': 0,
            'cache_saves': 0, 'evictions': 0, 'similarity_matches': 0
        }
        self._load_cache_from_file()

    def get(self, query: str) -> Optional[Dict]:
        """Retrieves a result from the cache, trying exact then similar matches."""
        self.stats['total_requests'] += 1
        if not self.config['enabled']:
            self.stats['cache_misses'] += 1
            return None

        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()

        if query_hash in self.cache:
            entry = self.cache[query_hash]
            if not entry.is_expired(self.config['ttl_seconds']):
                entry.update_access()
                self.stats['cache_hits'] += 1
                return {
                    'result': entry.result,
                    'model_used': entry.model_used,
                    'cached_at': entry.timestamp,
                    'access_count': entry.access_count,
                    'cache_hit': True,
                    'similarity_match': False
                }

        similar_result = self._find_similar_query(query)
        if similar_result:
            self.stats['cache_hits'] += 1
            self.stats['similarity_matches'] += 1
            return similar_result

        self.stats['cache_misses'] += 1
        return None

    def put(self, query: str, result: Dict, model_used: str):
        """Adds a new entry to the cache."""
        if not self.config['enabled']: return
        if len(self.cache) >= self.config['max_size']:
            self._evict_lru_entry()

        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        self.cache[query_hash] = CacheEntry(query, result, model_used)
        self.stats['cache_saves'] += 1

    def clear_cache(self) -> int:
        """
        Clears all items from the in-memory cache and returns the number of items removed.
        """
        cleared_count = len(self.cache)
        self.cache.clear()
        return cleared_count

    def _find_similar_query(self, query: str) -> Optional[Dict]:
        best_match, best_similarity = None, 0.0
        for entry in self.cache.values():
            if entry.is_expired(self.config['ttl_seconds']): continue

            similarity = self._calculate_similarity(query, entry.query)
            if similarity >= self.config['similarity_threshold'] and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match:
            best_match.update_access()
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
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        if not words1 or not words2: return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union

    def _evict_lru_entry(self):
        if not self.cache: return
        oldest_hash = min(self.cache, key=lambda h: self.cache[h].last_accessed)
        del self.cache[oldest_hash]
        self.stats['evictions'] += 1

    def save_cache_to_file(self):
        try:
            with open(FILE_PATHS['cache_file'], 'w', encoding='utf-8') as f:
                json.dump({h: e.to_dict() for h, e in self.cache.items()}, f, indent=2)
        except Exception as e:
            print(f"Error saving cache to file: {e}")

    def _load_cache_from_file(self):
        try:
            with open(FILE_PATHS['cache_file'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key, value in data.items():
                entry = CacheEntry.from_dict(value)
                if not entry.is_expired(self.config['ttl_seconds']):
                    self.cache[key] = entry
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def get_stats(self) -> Dict:
        total = max(self.stats['total_requests'], 1)
        stats_copy = self.stats.copy()
        stats_copy['hit_rate'] = stats_copy['cache_hits'] / total
        stats_copy['current_cache_size'] = len(self.cache)
        return stats_copy

    def get_cache_info(self) -> Dict:
        sorted_entries = sorted(self.cache.values(), key=lambda e: e.access_count, reverse=True)
        return {
            'total_entries': len(self.cache),
            'config': self.config,
            'entries': [e.to_dict() for e in sorted_entries[:10]]
        }

    def __del__(self):
        self.save_cache_to_file()