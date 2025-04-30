"""
Cache manager utility for storing and retrieving financial data.
"""

import os
import json
import time
import logging
import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from config.settings import CACHE_DIR, CACHE_SETTINGS
except ImportError:
    # Fallback settings if import fails
    CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_SETTINGS = {'ttl': 2592000, 'force_refresh': False}

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages caching of financial data to reduce API calls and handle rate limits.
    """
    
    def __init__(self, cache_name: str):
        """
        Initialize the cache manager.
        
        Args:
            cache_name (str): Name of the cache file (without extension)
        """
        self.cache_file = os.path.join(CACHE_DIR, f"{cache_name}_cache.json")
        self.ttl = CACHE_SETTINGS.get('ttl', 2592000)  # Default to 30 days
        self.force_refresh = CACHE_SETTINGS.get('force_refresh', False)
        # Maximum cache age is always 30 days (2592000 seconds)
        self.max_cache_age = 2592000
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """
        Load cache from disk if it exists and is valid.
        
        Returns:
            Dict: Loaded cache or empty cache if none exists
        """
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache has timestamp and is not expired
                if not self.force_refresh and 'timestamp' in cache_data:
                    cache_time = datetime.fromtimestamp(cache_data['timestamp'])
                    if datetime.now() - cache_time < timedelta(seconds=self.ttl):
                        logger.info(f"Using valid cache from {cache_time}")
                        return cache_data
                    else:
                        logger.info("Cache expired, will fetch fresh data")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
        
        # Return empty cache with current timestamp
        return {'timestamp': time.time(), 'data': {}}
    
    def save_cache(self) -> None:
        """
        Save the current cache to disk.
        """
        try:
            with open(self.cache_file, 'w') as f:
                self.cache['timestamp'] = time.time()
                json.dump(self.cache, f)
            logger.info(f"Saved cache with {len(self.cache['data'])} entries")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found
        """
        if key in self.cache['data']:
            entry = self.cache['data'][key]
            
            # Check entry timestamp if available 
            if isinstance(entry, dict) and 'timestamp' in entry and 'value' in entry:
                entry_time = datetime.fromtimestamp(entry['timestamp'])
                # Check if entry is older than 30 days
                if datetime.now() - entry_time > timedelta(seconds=self.max_cache_age):
                    logger.info(f"Cache entry for {key} is older than 30 days, will refresh")
                    return None
                value = entry['value']
            else:
                # For backward compatibility with older cache format
                value = entry
            
            # Convert JSON-serialized DataFrame back to DataFrame if needed
            if isinstance(value, str) and value.startswith('DataFrame:'):
                try:
                    return pd.read_json(value[10:])
                except Exception as e:
                    logger.warning(f"Error deserializing DataFrame for {key}: {e}")
                    return None
            
            return value
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
        """
        # Process the value for storage
        if isinstance(value, pd.DataFrame):
            processed_value = f"DataFrame:{value.to_json()}"
        else:
            processed_value = value
            
        # Store with timestamp for entry-level expiration
        self.cache['data'][key] = {
            'timestamp': time.time(),
            'value': processed_value
        }
        
        # Periodically save cache to disk (10% chance)
        if hash(key) % 10 == 0:
            self.save_cache()
    
    def has(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key exists, False otherwise
        """
        return key in self.cache['data']
    
    def clear(self) -> None:
        """
        Clear the entire cache.
        """
        self.cache = {'timestamp': time.time(), 'data': {}}
        self.save_cache()
    
    def delete(self, key: str) -> None:
        """
        Delete a specific key from the cache.
        
        Args:
            key (str): Cache key to delete
        """
        if key in self.cache['data']:
            del self.cache['data'][key]
    
    def get_age(self) -> int:
        """
        Get the age of the cache in seconds.
        
        Returns:
            int: Age in seconds
        """
        if 'timestamp' in self.cache:
            return int(time.time() - self.cache['timestamp'])
        return 0


# Create a rate limiter to manage API request timing
class RateLimiter:
    """
    Manages rate limiting for API requests.
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests (int): Maximum requests allowed
            time_window (int): Time window in seconds (default: 1 hour)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = []
        self.limited = False
    
    def check_limit(self) -> bool:
        """
        Check if we're currently rate limited.
        
        Returns:
            bool: True if rate limited, False otherwise
        """
        current_time = time.time()
        
        # Remove timestamps older than the time window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts <= self.time_window]
        
        # Check if we've hit the limit
        self.limited = len(self.request_timestamps) >= self.max_requests
        return self.limited
    
    def add_request(self) -> None:
        """
        Record a new request.
        """
        self.request_timestamps.append(time.time())
    
    def wait_time(self) -> float:
        """
        Calculate how long to wait before making another request.
        
        Returns:
            float: Time to wait in seconds
        """
        if not self.request_timestamps:
            return 0
            
        current_time = time.time()
        
        if not self.check_limit():
            return 0
            
        # Calculate when the oldest request will expire
        oldest = min(self.request_timestamps)
        wait_seconds = oldest + self.time_window - current_time
        
        return max(0, wait_seconds) 