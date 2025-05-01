"""
Financial data service for retrieving market data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import random
import os
import sys
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from curl_cffi import requests

# Add parent directory to path to enable relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_manager import CacheManager, RateLimiter
from config.settings import API_SETTINGS, ASSETS

logger = logging.getLogger(__name__)

class FinancialDataService:
    """
    Service for retrieving financial data with caching and rate limiting.
    Implements retry logic when rate limited.
    """
    
    def __init__(self):
        """Initialize the financial data service."""
        self.cache = CacheManager("financial_data")
        self.rate_limiter = RateLimiter(
            max_requests=API_SETTINGS['yfinance']['max_requests_per_hour'],
            time_window=3600
        )
        # Max retry parameters
        self.max_retries = 3
        self.retry_backoff_factor = 2.0
        # Create a session with curl_cffi
        self.session = requests.Session(impersonate="chrome")
    
    def _respect_rate_limit(self) -> bool:
        """
        Check rate limits and wait if necessary.
        
        Returns:
            bool: True if rate limit allows request, False if limited and retries exhausted
        """
        # Add a small delay to avoid hitting rate limits
        delay = API_SETTINGS['yfinance']['request_delay']
        time.sleep(delay + random.random() * delay)
        
        # Check if we're rate limited
        if self.rate_limiter.check_limit():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.warning(f"Rate limited, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            return False
        
        # Record the request
        self.rate_limiter.add_request()
        return True
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry a function with exponential backoff when rate limited.
        
        Args:
            func: The function to retry
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                if retries > 0:
                    # Add exponential backoff delay
                    delay = (self.retry_backoff_factor ** retries) * (1 + random.random())
                    logger.info(f"Retry attempt {retries}/{self.max_retries}, waiting {delay:.2f} seconds")
                    time.sleep(delay)
                
                # Check rate limit before making the call
                if not self._respect_rate_limit() and retries < self.max_retries:
                    retries += 1
                    continue
                
                # Call the function
                result = func(*args, **kwargs)
                
                # If we get here, the call was successful
                if retries > 0:
                    logger.info(f"Successfully retrieved data after {retries} retries")
                return result
            
            except Exception as e:
                if "Too Many Requests" in str(e) and retries < self.max_retries:
                    retries += 1
                    # Add extra delay for rate limiting
                    delay = (self.retry_backoff_factor ** retries) * (5 + random.random() * 5)
                    logger.warning(f"Rate limit exceeded, retry {retries}/{self.max_retries} after {delay:.2f}s delay")
                    time.sleep(delay)
                else:
                    # Re-raise the exception if it's not rate-limiting or we've exhausted retries
                    logger.error(f"Error after {retries} retries: {str(e)}")
                    raise
        
        # If we get here, all retries were exhausted
        raise Exception(f"Maximum retries ({self.max_retries}) exceeded")
    
    def _check_cache(self, cache_key: str, data_type: Any = None) -> Optional[Any]:
        """
        Check if data exists in cache and is of the expected type.
        
        Args:
            cache_key (str): Cache key to check
            data_type (Any, optional): Expected data type
            
        Returns:
            Optional[Any]: Cached data if valid, None otherwise
        """
        cached_data = self.cache.get(cache_key)
        
        # Type-specific validation
        if cached_data is not None:
            if data_type is pd.DataFrame and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                logger.info(f"Using cached DataFrame for {cache_key}")
                return cached_data
            elif data_type is dict and isinstance(cached_data, dict) and cached_data:
                logger.info(f"Using cached dict for {cache_key}")
                return cached_data
            elif data_type is list and isinstance(cached_data, list):
                logger.info(f"Using cached list for {cache_key}")
                return cached_data
            elif data_type in (float, int) and isinstance(cached_data, (float, int)):
                logger.info(f"Using cached value for {cache_key}")
                return cached_data
            elif data_type is None:
                logger.info(f"Using cached data for {cache_key}")
                return cached_data
                
        return None
    
    def _validate_symbols(self, symbols: List[str], history_period: str = "5d", validation_func: Optional[Callable] = None) -> List[str]:
        """
        Validate a list of symbols by checking if they have valid data.
        
        Args:
            symbols (List[str]): List of symbols to validate
            history_period (str): Period to check for historical data
            validation_func (Callable, optional): Custom validation function
            
        Returns:
            List[str]: List of valid symbols
        """
        valid_symbols = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol, session=self.session)
                
                # If custom validation function provided, use it
                if validation_func and callable(validation_func):
                    if validation_func(ticker):
                        valid_symbols.append(symbol)
                # Otherwise use default validation (check history)
                else:
                    hist = ticker.history(period=history_period)
                    if not hist.empty:
                        valid_symbols.append(symbol)
            except Exception as e:
                logger.warning(f"Error validating symbol {symbol}: {str(e)}")
                continue
                
        return valid_symbols
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data for a symbol with caching.
        Uses retry logic when rate limited.
        
        Args:
            symbol (str): Symbol to fetch data for
            period (str): Time period (e.g., "1d", "1mo", "1y")
            
        Returns:
            pd.DataFrame: Historical price data
        """
        cache_key = f"historical_{symbol}_{period}"
        
        # Check cache first
        cached_data = self._check_cache(cache_key, pd.DataFrame)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, fetch from API with retries
        try:
            def fetch_data():
                ticker = yf.Ticker(symbol, session=self.session)
                history = ticker.history(period=period)
                
                if history.empty:
                    raise ValueError(f"No historical data found for {symbol}")
                
                return history
            
            # Fetch with retry logic
            history = self._retry_with_backoff(fetch_data)
            
            # Cache the result
            self.cache.set(cache_key, history)
            return history
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Return empty DataFrame
            return pd.DataFrame()
    
    def get_ticker_info(self, symbol: str) -> Dict:
        """
        Get ticker information with caching.
        Uses retry logic when rate limited.
        
        Args:
            symbol (str): Symbol to fetch info for
            
        Returns:
            Dict: Ticker information
        """
        cache_key = f"info_{symbol}"
        
        # Check cache first
        cached_data = self._check_cache(cache_key, dict)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, fetch from API with retries
        try:
            def fetch_info():
                ticker = yf.Ticker(symbol, session=self.session)
                info = ticker.info
                
                if not info:
                    raise ValueError(f"No info found for {symbol}")
                
                return info
            
            # Fetch with retry logic
            info = self._retry_with_backoff(fetch_info)
            
            # Cache the result
            self.cache.set(cache_key, info)
            return info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            # Return empty dict
            return {}
    
    def get_sector_stocks(self) -> Dict[str, List[str]]:
        """
        Get the list of stocks by sector with caching.
        Uses retry logic when rate limited.
        
        Returns:
            Dict[str, List[str]]: Dictionary of sectors and their stock symbols
        """
        cache_key = "sector_stocks"
        
        # Check cache first
        cached_data = self._check_cache(cache_key, dict)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, use the predefined sectors from settings
        try:
            def fetch_sectors():
                # Use predefined sectors from settings
                predefined_sectors = ASSETS['stocks']['sectors']
                
                # Validate which symbols are available
                validated_sectors = {}
                for sector, symbols in predefined_sectors.items():
                    # Take only a few stocks from each sector to minimize API calls
                    sample_symbols = symbols[:3]  # Take up to 3 symbols per sector
                    
                    # Validate symbols
                    valid_symbols = self._validate_symbols(
                        sample_symbols, 
                        validation_func=lambda ticker: ticker.info and 'regularMarketPrice' in ticker.info
                    )
                    
                    if valid_symbols:
                        validated_sectors[sector] = valid_symbols
                
                if not validated_sectors:
                    raise ValueError("No valid sector stocks found")
                
                return validated_sectors
            
            # Fetch with retry logic
            sectors = self._retry_with_backoff(fetch_sectors)
            
            # Cache the result
            self.cache.set(cache_key, sectors)
            return sectors
            
        except Exception as e:
            logger.error(f"Error getting sector stocks: {str(e)}")
            # Return the predefined sectors from settings
            return ASSETS['stocks']['sectors']
    
    def get_crypto_list(self) -> List[str]:
        """
        Get a list of supported cryptocurrencies with caching.
        
        Returns:
            List[str]: List of cryptocurrency symbols
        """
        # Use predefined list to avoid excessive API calls
        return ASSETS['crypto']['fallback']
    
    def get_etfs_by_category(self, category: str, duration: str = None) -> List[str]:
        """
        Generic method to get ETFs by category with caching.
        Uses retry logic when rate limited.
        
        Args:
            category (str): Category of ETFs ('bonds', 'gold', etc.)
            duration (str, optional): Duration for bond ETFs
            
        Returns:
            List[str]: List of ETF symbols
        """
        # Determine cache key and ETF list based on category
        if category == 'bonds':
            if not duration or duration == 'all':
                etfs = (ASSETS['bonds']['short_term'] + 
                      ASSETS['bonds']['intermediate'] + 
                      ASSETS['bonds']['long_term'])
                cache_key = f"bond_etfs_all"
            else:
                etfs = ASSETS['bonds'][f'{duration}_term']
                cache_key = f"bond_etfs_{duration}"
        elif category == 'gold':
            etfs = ASSETS['gold']['etfs']
            cache_key = "gold_etfs"
        else:
            logger.error(f"Unknown ETF category: {category}")
            return []
        
        # Check cache first
        cached_data = self._check_cache(cache_key, list)
        if cached_data is not None:
            return cached_data
        
        # Validate which ones are available with retry logic
        try:
            def validate_etfs():
                valid_etfs = self._validate_symbols(etfs)
                
                if not valid_etfs:
                    raise ValueError(f"No valid {category} ETFs found")
                
                return valid_etfs
            
            # Fetch with retry logic
            valid_etfs = self._retry_with_backoff(validate_etfs)
            
            # Cache the result
            self.cache.set(cache_key, valid_etfs)
            return valid_etfs
            
        except Exception as e:
            logger.error(f"Error validating {category} ETFs: {str(e)}")
            # Return the predefined ETFs
            return etfs
    
    def get_bond_etfs(self, duration: str = "all") -> List[str]:
        """
        Get a list of bond ETFs based on duration with caching.
        Uses retry logic when rate limited.
        
        Args:
            duration (str): Duration category ('short', 'intermediate', 'long', or 'all')
            
        Returns:
            List[str]: List of bond ETF symbols
        """
        return self.get_etfs_by_category('bonds', duration)
    
    def get_gold_etfs(self) -> List[str]:
        """
        Get a list of gold ETFs with caching.
        Uses retry logic when rate limited.
        
        Returns:
            List[str]: List of gold ETF symbols
        """
        return self.get_etfs_by_category('gold')
    
    def get_treasury_rate(self, duration: str = "short") -> float:
        """
        Get current Treasury rate with caching.
        Uses retry logic when rate limited.
        
        Args:
            duration (str): 'short' for 3-month or 'long' for 10-year
            
        Returns:
            float: Current Treasury rate (decimal)
        """
        symbol = ASSETS['bonds']['treasury_rates'].get(duration, '^IRX')
        cache_key = f"treasury_{duration}_rate"
        
        # Check cache first
        cached_data = self._check_cache(cache_key, float)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, fetch from API with retries
        try:
            def fetch_rate():
                ticker = yf.Ticker(symbol, session=self.session)
                hist = ticker.history(period="1d")
                
                if hist.empty:
                    raise ValueError(f"No Treasury rate data found for {symbol}")
                
                # Get rate and convert to decimal
                rate = hist['Close'].iloc[-1] / 100
                return rate
            
            # Fetch with retry logic
            rate = self._retry_with_backoff(fetch_rate)
            
            # Cache the result
            self.cache.set(cache_key, rate)
            return rate
            
        except Exception as e:
            logger.error(f"Error fetching Treasury rate for {symbol}: {str(e)}")
            # Return a reasonable default value
            return 0.02  # 2% as a conservative default
    
    def calculate_metrics(self, symbol: str) -> Dict:
        """
        Calculate various metrics for a symbol based on historical data with caching.
        Uses retry logic when rate limited.
        
        Args:
            symbol (str): Symbol to calculate metrics for
            
        Returns:
            Dict: Calculated metrics
        """
        cache_key = f"metrics_{symbol}"
        
        # Check cache first
        cached_data = self._check_cache(cache_key, dict)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, calculate metrics with retries
        try:
            def compute_metrics():
                # Get historical data and ticker info
                ticker = yf.Ticker(symbol, session=self.session)
                hist = ticker.history(period="1y")
                info = ticker.info
                
                if hist.empty:
                    raise ValueError(f"No historical data for metrics calculation: {symbol}")
                
                # Calculate basic metrics
                returns = hist['Close'].pct_change().dropna()
                current_price = hist['Close'].iloc[-1]
                
                metrics = {
                    'current_price': current_price,
                    'momentum_1m': (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) if len(hist) >= 21 else 0,
                    'momentum_3m': (hist['Close'].iloc[-1] / hist['Close'].iloc[-63] - 1) if len(hist) >= 63 else 0, 
                    'momentum_6m': (hist['Close'].iloc[-1] / hist['Close'].iloc[-126] - 1) if len(hist) >= 126 else 0,
                    'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
                    'average_volume': hist['Volume'].mean(),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0),
                    'pe_ratio': info.get('forwardPE', 0) or info.get('trailingPE', 0) or 0,
                    'dividend_yield': info.get('dividendYield', 0) or 0,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    # Use a more deterministic predicted return based on actual data
                    'predicted_return': max(0.05, min(0.15, returns.mean() * 252))  # Annualized mean return, capped between 5-15%
                }
                
                return metrics
            
            # Calculate metrics with retry logic
            metrics = self._retry_with_backoff(compute_metrics)
            
            # Cache the result
            self.cache.set(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            # Return minimal metrics with zeros
            return {
                'current_price': 0,
                'momentum_1m': 0,
                'momentum_3m': 0,
                'momentum_6m': 0,
                'volatility': 0,
                'average_volume': 0,
                'market_cap': 0,
                'beta': 1.0,
                'pe_ratio': 0,
                'dividend_yield': 0,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'predicted_return': 0.05
            }
    
    def get_market_sentiment(self) -> Dict:
        """
        Get current market sentiment metrics with caching.
        Uses retry logic when rate limited.
        
        Returns:
            Dict: Market sentiment metrics
        """
        cache_key = "market_sentiment"
        
        # Check cache first
        cached_data = self._check_cache(cache_key, dict)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, fetch from API with retries
        try:
            def fetch_sentiment():
                # Get current values for major indices
                indices = {
                    'S&P 500': '^GSPC',
                    'Dow Jones': '^DJI',
                    'NASDAQ': '^IXIC',
                    'VIX': '^VIX'
                }
                
                sentiment_data = {}
                
                for name, symbol in indices.items():
                    ticker = yf.Ticker(symbol, session=self.session)
                    history = ticker.history(period="1mo")
                    
                    if history.empty:
                        continue
                    
                    # Current and previous values
                    current = history['Close'].iloc[-1]
                    prev = history['Close'].iloc[-2] if len(history) > 1 else None
                    
                    # Add to sentiment data
                    sentiment_data[name] = {
                        'current': current,
                        'previous': prev,
                        'change': (current - prev) / prev * 100 if prev else 0,
                        'trend': self._calculate_trend(history)
                    }
                
                if not sentiment_data:
                    raise ValueError("No sentiment data could be retrieved")
                
                # Calculate overall market sentiment
                sentiment_data['Overall'] = self._calculate_overall_sentiment(sentiment_data)
                
                return sentiment_data
            
            # Fetch with retry logic
            sentiment = self._retry_with_backoff(fetch_sentiment)
            
            # Cache the result
            self.cache.set(cache_key, sentiment)
            return sentiment
            
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {str(e)}")
            # Return empty dict
            return {}
    
    def _calculate_trend(self, history: pd.DataFrame) -> str:
        """Calculate trend from historical data."""
        if history.empty or len(history) < 5:
            return "neutral"
        
        # Get last 5 closing prices
        closes = history['Close'].tail(5)
        
        # Simple trend calculation
        if closes.iloc[-1] > closes.iloc[0] * 1.05:
            return "strongly_bullish"
        elif closes.iloc[-1] > closes.iloc[0] * 1.02:
            return "bullish"
        elif closes.iloc[-1] < closes.iloc[0] * 0.95:
            return "strongly_bearish"
        elif closes.iloc[-1] < closes.iloc[0] * 0.98:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_overall_sentiment(self, sentiment_data: Dict) -> Dict:
        """Calculate overall market sentiment."""
        # Initialize counters for different sentiments
        sentiment_counts = {
            'strongly_bullish': 0,
            'bullish': 0,
            'neutral': 0,
            'bearish': 0,
            'strongly_bearish': 0
        }
        
        # Count trends
        for key, data in sentiment_data.items():
            if key == 'Overall':
                continue
            
            if 'trend' in data:
                sentiment_counts[data['trend']] += 1
        
        # Determine overall trend
        total = sum(sentiment_counts.values())
        if total == 0:
            overall_trend = "neutral"
        else:
            # Weighted calculation
            score = (
                sentiment_counts['strongly_bullish'] * 2 +
                sentiment_counts['bullish'] * 1 +
                sentiment_counts['neutral'] * 0 +
                sentiment_counts['bearish'] * -1 +
                sentiment_counts['strongly_bearish'] * -2
            ) / total
            
            if score > 1.0:
                overall_trend = "strongly_bullish"
            elif score > 0.3:
                overall_trend = "bullish"
            elif score < -1.0:
                overall_trend = "strongly_bearish"
            elif score < -0.3:
                overall_trend = "bearish"
            else:
                overall_trend = "neutral"
        
        # Calculate average change
        changes = [data['change'] for key, data in sentiment_data.items() if key != 'Overall' and 'change' in data]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        return {
            'trend': overall_trend,
            'change': avg_change
        }
    
    def clear_cache(self) -> None:
        """
        Clear the entire data cache.
        """
        self.cache.clear()
        logger.info("Cleared financial data cache")

# Create a singleton instance
financial_data_service = FinancialDataService() 