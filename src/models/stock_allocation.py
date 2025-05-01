import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import time
import random
import os
import json
import traceback
from curl_cffi import requests

from src.config.settings import RISK_PROFILES, ASSETS
from src.utils.cache_utils import CACHE_DIR

logger = logging.getLogger(__name__)

# Create a cache directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

class StockAllocationModel:
    """
    Stock allocation model that selects stocks based on machine learning predictions
    and fundamental analysis.
    """
    
    def __init__(self):
        """Initialize the stock allocation model."""
        self.metrics = {
            'pe_ratio': {'weight': 0.15, 'lower_better': True},
            'dividend_yield': {'weight': 0.10, 'lower_better': False},
            'beta': {'weight': 0.10, 'lower_better': True},
            'momentum': {'weight': 0.15, 'lower_better': False},
            'volatility': {'weight': 0.10, 'lower_better': True},
            'volume': {'weight': 0.10, 'lower_better': False},
            'predicted_return': {'weight': 0.30, 'lower_better': False}  # ML prediction gets highest weight
        }
        self.models = {}  # Cache for trained models
        self.cache = {}   # Cache for stock data
        self.rate_limited = False
        self.req_count = 0
        self.last_req_time = datetime.now()
        
        # Load cached data if available
        self._load_cache()
        
        # Create curl_cffi session
        self.session = requests.Session(impersonate="chrome")
    
    def _get_sector_stocks(self) -> Dict[str, List[str]]:
        """
        Get a curated list of stocks by sector.
        
        Returns:
            Dict[str, List[str]]: Stocks grouped by sector
        """
        try:
            # Use the predefined sectors from settings.py
            predefined_sectors = ASSETS['stocks']['sectors']
            
            # If we're rate limited, return a much smaller set to reduce API calls
            if self.rate_limited:
                logger.warning("Using reduced stock set due to rate limiting")
                reduced_sectors = {}
                # Take only the first stock from each sector
                for sector, symbols in predefined_sectors.items():
                    reduced_sectors[sector] = [symbols[0]]
                return reduced_sectors
            
            # Validate the symbols to ensure they're tradeable
            validated_sectors = {}
            for sector, symbols in predefined_sectors.items():
                valid_symbols = []
                for symbol in symbols:
                    try:
                        # First check if we have cached data
                        if symbol in self.cache:
                            valid_symbols.append(symbol)
                            continue
                            
                        # Apply rate limiting
                        self._respect_rate_limit()
                        
                        ticker = yf.Ticker(symbol, session=self.session)
                        # Do a simple validation by getting a small amount of history
                        hist = ticker.history(period="5d")
                        if not hist.empty:
                            valid_symbols.append(symbol)
                    except Exception as e:
                        if "Too Many Requests" in str(e):
                            self.rate_limited = True
                            logger.warning(f"Rate limited detected, will use reduced dataset")
                            # Return immediately with reduced set
                            reduced_sectors = {}
                            for s, syms in predefined_sectors.items():
                                reduced_sectors[s] = [syms[0]]
                            return reduced_sectors
                        else:
                            logger.warning(f"Skipping {symbol} due to error: {str(e)}")
                if valid_symbols:
                    validated_sectors[sector] = valid_symbols
                    
            return validated_sectors
                
        except Exception as e:
            logger.error(f"Error getting sector stocks: {str(e)}")
            return {}
    
    def _respect_rate_limit(self):
        """
        Respect rate limits by adding delays between requests.
        """
        self.req_count += 1
        now = datetime.now()
        
        # If we've made too many requests in a short time, slow down
        if self.req_count > 50:  # Reset counter after 50 requests
            self.req_count = 0
            time.sleep(2)  # 2 second pause after every 50 requests
        
        # Add a small delay between each request
        time_since_last = (now - self.last_req_time).total_seconds()
        if time_since_last < 0.2:  # Ensure at least 0.2 seconds between requests
            delay = 0.2 - time_since_last
            time.sleep(delay)
            
        self.last_req_time = datetime.now()
    
    def _save_cache(self):
        """
        Save the cache to disk.
        """
        cache_file = os.path.join(CACHE_DIR, 'stock_data_cache.json')
        try:
            serializable_cache = {}
            for symbol, data in self.cache.items():
                if isinstance(data, pd.DataFrame):
                    serializable_cache[symbol] = data.to_json()
                else:
                    serializable_cache[symbol] = data
                    
            with open(cache_file, 'w') as f:
                json.dump(serializable_cache, f)
                
            logger.info(f"Saved stock data cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _load_cache(self):
        """
        Load the cache from disk.
        """
        cache_file = os.path.join(CACHE_DIR, 'stock_data_cache.json')
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    serialized_cache = json.load(f)
                
                for symbol, data in serialized_cache.items():
                    try:
                        if isinstance(data, str):
                            self.cache[symbol] = pd.read_json(data)
                        else:
                            self.cache[symbol] = data
                    except Exception as e:
                        logger.warning(f"Error deserializing cache for {symbol}: {str(e)}")
                
                logger.info(f"Loaded stock data cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
    
    def _get_historical_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Get historical price data for a stock.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period for data
            
        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            # Check if we have cached data
            cache_key = f"{symbol}_{period}"
            if cache_key in self.cache:
                return self.cache[cache_key]
                
            # Apply rate limiting
            self._respect_rate_limit()
            
            ticker = yf.Ticker(symbol, session=self.session)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No historical data for {symbol}")
                return pd.DataFrame()
                
            # Cache the data
            self.cache[cache_key] = hist
            
            # Periodically save the cache
            if random.random() < 0.1:  # 10% chance to save on each request
                self._save_cache()
                
            return hist
        except Exception as e:
            if "Too Many Requests" in str(e):
                self.rate_limited = True
                logger.error(f"Rate limited when getting data for {symbol}")
            else:
                logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning model.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: Feature dataframe
        """
        if data.empty:
            return pd.DataFrame()
            
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Basic price features
            df['Returns'] = df['Close'].pct_change()
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Ratio features
            df['MA5_Ratio'] = df['Close'] / df['MA5']
            df['MA10_Ratio'] = df['Close'] / df['MA10']
            df['MA20_Ratio'] = df['Close'] / df['MA20']
            df['MA50_Ratio'] = df['Close'] / df['MA50']
            
            # Volatility features
            df['Volatility5'] = df['Returns'].rolling(window=5).std()
            df['Volatility10'] = df['Returns'].rolling(window=10).std()
            df['Volatility20'] = df['Returns'].rolling(window=20).std()
            
            # Volume features
            df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
            
            # Momentum features
            df['Momentum5'] = df['Close'].pct_change(periods=5)
            df['Momentum10'] = df['Close'].pct_change(periods=10)
            df['Momentum20'] = df['Close'].pct_change(periods=20)
            
            # Target variable - 30-day future return
            df['Target'] = df['Close'].pct_change(periods=30).shift(-30)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()
    
    def _train_model(self, features: pd.DataFrame, symbol: str) -> tuple:
        """
        Train machine learning model to predict future returns.
        
        Args:
            features (pd.DataFrame): Feature dataframe
            symbol (str): Stock symbol
            
        Returns:
            tuple: (model, scaler, feature_names, model_quality)
        """
        if features.empty:
            return None, None, None, 0
        
        try:
            # Define features and target
            feature_names = [
                'Returns', 'MA5_Ratio', 'MA10_Ratio', 'MA20_Ratio', 'MA50_Ratio',
                'Volatility5', 'Volatility10', 'Volatility20',
                'Volume_Ratio', 'Momentum5', 'Momentum10', 'Momentum20'
            ]
            X = features[feature_names]
            y = features['Target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            model_quality = test_score  # Use R² as model quality
            
            logger.info(f"Trained model for {symbol} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
            
            return model, scaler, feature_names, model_quality
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return None, None, None, 0
    
    def _predict_return(self, symbol: str) -> float:
        """
        Predict future return for a stock using machine learning.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Predicted return
        """
        try:
            # Get historical data
            hist_data = self._get_historical_data(symbol)
            if hist_data.empty:
                return 0.0
                
            # Prepare features
            features = self._prepare_features(hist_data)
            if features.empty:
                return 0.0
                
            # Train model if not cached
            if symbol not in self.models:
                model, scaler, feature_names, model_quality = self._train_model(features, symbol)
                if model is None:
                    return 0.0
                self.models[symbol] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'quality': model_quality
                }
            
            # Get current features
            current_features = features.iloc[-1:][self.models[symbol]['feature_names']]
            
            # Scale and predict
            scaled_features = self.models[symbol]['scaler'].transform(current_features)
            predicted_return = self.models[symbol]['model'].predict(scaled_features)[0]
            
            logger.info(f"Predicted return for {symbol}: {predicted_return:.4f}")
            
            return predicted_return
            
        except Exception as e:
            logger.error(f"Error predicting return for {symbol}: {str(e)}")
            return 0.0
    
    def _get_stock_data(self, symbol: str) -> Dict:
        """
        Get stock data and metrics for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Stock metrics and data
        """
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                return {}
            
            # Calculate metrics
            returns = hist['Close'].pct_change()
            momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
            volatility = returns.std() * np.sqrt(252)
            volume = hist['Volume'].mean()
            
            # Predict future return using ML
            predicted_return = self._predict_return(symbol)
            
            return {
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'momentum': momentum,
                'volatility': volatility,
                'volume': volume,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'predicted_return': predicted_return
            }
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return {}
    
    def _normalize_metric(self, value: float, lower_better: bool) -> float:
        """
        Normalize a metric value to a score between 0 and 1.
        
        Args:
            value (float): Metric value
            lower_better (bool): Whether lower values are better
            
        Returns:
            float: Normalized score
        """
        if lower_better:
            return 1 / (1 + value)
        return value / (1 + value)
    
    def _calculate_stock_score(self, metrics: Dict) -> float:
        """
        Calculate overall score for a stock based on its metrics.
        
        Args:
            metrics (Dict): Stock metrics
            
        Returns:
            float: Overall score
        """
        if not metrics:
            return 0.0
        
        score = 0.0
        for metric, config in self.metrics.items():
            if metric in metrics:
                normalized_value = self._normalize_metric(
                    metrics[metric],
                    config['lower_better']
                )
                score += normalized_value * config['weight']
        
        return score
    
    def _select_top_stocks(self, amount: float) -> Dict[str, float]:
        """
        Select top stocks based on scoring and allocate investment amount.
        
        Args:
            amount (float): Total amount to invest
            
        Returns:
            Dict[str, float]: Stock allocations
        """
        try:
            # Get sector stocks
            sectors = self._get_sector_stocks()
            if not sectors:
                logger.error("No sector stocks available")
                return {}
            
            # Get and score all stocks
            stock_scores = {}
            for sector, symbols in sectors.items():
                for symbol in symbols:
                    metrics = self._get_stock_data(symbol)
                    if metrics:
                        score = self._calculate_stock_score(metrics)
                        stock_scores[symbol] = {
                            'score': score,
                            'metrics': metrics
                        }
            
            # Sort stocks by score
            sorted_stocks = sorted(
                stock_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            # Select top stocks (up to 5)
            top_stocks = sorted_stocks[:5]
            
            # Calculate allocations based on scores
            total_score = sum(stock['score'] for _, stock in top_stocks)
            if total_score == 0:
                return {}
            
            allocations = {}
            for symbol, data in top_stocks:
                allocation = (data['score'] / total_score) * amount
                allocations[symbol] = allocation
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error selecting stocks: {str(e)}")
            return {}
    
    def _calculate_metrics(self, symbol: str, hist_data: pd.DataFrame) -> Dict:
        """
        Calculate metrics for a stock based on historical data.
        
        Args:
            symbol (str): Stock symbol
            hist_data (pd.DataFrame): Historical price data
            
        Returns:
            Dict: Stock metrics
        """
        try:
            # Fetch ticker info
            ticker = yf.Ticker(symbol, session=self.session)
            info = ticker.info
            
            # Calculate return metrics
            returns = hist_data['Close'].pct_change()
            momentum = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0] - 1)
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Get fundamental metrics
            pe_ratio = info.get('forwardPE', 0) or info.get('trailingPE', 0) or 0
            dividend_yield = info.get('dividendYield', 0) or 0
            beta = info.get('beta', 1.0) or 1.0
            volume = hist_data['Volume'].mean()
            
            return {
                'pe_ratio': pe_ratio,
                'dividend_yield': dividend_yield,
                'beta': beta,
                'momentum': momentum,
                'volatility': volatility,
                'volume': volume,
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            return {}
    
    def _calculate_score(self, metrics: Dict) -> float:
        """
        Calculate overall score for a stock based on its metrics.
        
        Args:
            metrics (Dict): Stock metrics
            
        Returns:
            float: Overall score
        """
        if not metrics:
            return 0.0
        
        try:
            score = 0.0
            total_weight = 0.0
            
            for metric, config in self.metrics.items():
                if metric in metrics:
                    normalized_value = self._normalize_metric(
                        metrics[metric],
                        config['lower_better']
                    )
                    score += normalized_value * config['weight']
                    total_weight += config['weight']
            
            # Normalize score if we didn't have all metrics
            if total_weight > 0:
                score = score * (sum(m['weight'] for m in self.metrics.values()) / total_weight)
            
            return score
        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            return 0.0
    
    def _calculate_portfolio_metrics(self, allocations, stock_metrics=None):
        """
        Calculate overall portfolio metrics based on individual stock metrics.
        
        Args:
            allocations (dict): Dictionary of stock allocations
            stock_metrics (dict): Dictionary of stock metrics
            
        Returns:
            dict: Portfolio metrics
        """
        try:
            logger.info(f"Calculating portfolio metrics for {len(allocations)} stocks")
            
            total_amount = sum(allocations.values())
            if total_amount == 0:
                logger.warning("Total allocation amount is zero")
                return {
                    'predicted_return': 0.05,  # Default 5% return
                    'dividend_yield': 0.02,
                    'beta': 1.0,
                    'pe_ratio': 15,
                    'diversification_score': 0.5
                }
            
            # If stock_metrics is None or empty, create it
            if stock_metrics is None or not stock_metrics:
                logger.info("No stock metrics provided, generating them")
                stock_metrics = {}
                for symbol in allocations.keys():
                    try:
                        hist_data = self._get_historical_data(symbol)
                        if not hist_data.empty:
                            stock_metrics[symbol] = self._calculate_metrics(symbol, hist_data)
                    except Exception as e:
                        logger.warning(f"Could not calculate metrics for {symbol}: {str(e)}")
            
            # Calculate weighted metrics
            weighted_predicted_return = 0
            weighted_dividend_yield = 0
            weighted_beta = 0
            weighted_pe = 0
            
            for symbol, amount in allocations.items():
                weight = amount / total_amount
                
                # Try to get metrics for this symbol
                metrics = None
                if stock_metrics and isinstance(stock_metrics, dict):
                    # Try different ways to access the metrics
                    if symbol in stock_metrics:
                        metrics = stock_metrics[symbol]
                    elif isinstance(stock_metrics.get(symbol), dict):
                        metrics = stock_metrics.get(symbol)
                
                # If we couldn't find metrics, generate default ones
                if not metrics:
                    logger.warning(f"No metrics found for {symbol}, using defaults")
                    metrics = {
                        'predicted_return': 0.07,  # 7% default return
                        'dividend_yield': 0.02,    # 2% default yield
                        'beta': 1.0,               # Market beta
                        'pe_ratio': 15             # Average P/E ratio
                    }
                
                # Add to weighted metrics
                weighted_predicted_return += metrics.get('predicted_return', 0.07) * weight
                weighted_dividend_yield += metrics.get('dividend_yield', 0.02) * weight
                weighted_beta += metrics.get('beta', 1.0) * weight
                weighted_pe += metrics.get('pe_ratio', 15) * weight
            
            # Calculate sector exposure
            sector_exposure = {}
            for symbol, amount in allocations.items():
                sector = "Unknown"
                if stock_metrics and symbol in stock_metrics and 'sector' in stock_metrics[symbol]:
                    sector = stock_metrics[symbol]['sector']
                
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += amount / total_amount
            
            # Calculate diversification score (0-1)
            num_stocks = len(allocations)
            sector_count = len(sector_exposure)
            diversification_score = min(1.0, (num_stocks / 20) * 0.5 + (sector_count / 8) * 0.5)
            
            # Portfolio summary metrics
            result = {
                'predicted_return': weighted_predicted_return,
                'dividend_yield': weighted_dividend_yield,
                'beta': weighted_beta,
                'pe_ratio': weighted_pe,
                'diversification_score': diversification_score,
                'sector_exposure': sector_exposure
            }
            
            logger.info(f"Calculated portfolio metrics successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            # Return fallback metrics
            return {
                'predicted_return': 0.05,  # Default 5% return
                'dividend_yield': 0.02,
                'beta': 1.0,
                'pe_ratio': 15,
                'diversification_score': 0.5,
                'sector_exposure': {'Technology': 0.5, 'Other': 0.5}
            }
    
    def run(self, amount: float, risk_profile: str) -> Dict:
        """
        Run the stock allocation model.
        
        Args:
            amount (float): Total investment amount
            risk_profile (str): Risk profile (conservative, moderate, aggressive)
            
        Returns:
            Dict: Stock allocation results
        """
        # If we've detected rate limiting, use cached/fallback data when possible
        if self.rate_limited:
            logger.warning("Using fallback data due to rate limiting")
        
        try:
            # Get stocks by sector
            sector_stocks = self._get_sector_stocks()
            
            # Flatten stock list
            all_stocks = []
            for stocks in sector_stocks.values():
                all_stocks.extend(stocks)
                
            if not all_stocks:
                logger.error("No valid stocks found in any sector")
                
                # Provide fallback allocations if we're rate limited
                if self.rate_limited:
                    fallback_allocations = {
                        'AAPL': amount * 0.2,  # Apple
                        'MSFT': amount * 0.2,  # Microsoft
                        'AMZN': amount * 0.15, # Amazon
                        'GOOGL': amount * 0.15, # Google
                        'BRK-B': amount * 0.1, # Berkshire Hathaway
                        'JNJ': amount * 0.1,   # Johnson & Johnson
                        'PG': amount * 0.05,   # Procter & Gamble
                        'V': amount * 0.05     # Visa
                    }
                    
                    logger.info(f"Using fallback allocations for {len(fallback_allocations)} stocks")
                    
                    return {
                        'allocations': fallback_allocations,
                        'metrics': {
                            'predicted_return': 0.08,  # 8% expected return
                            'diversification_score': 0.7,
                            'sector_exposure': {
                                'Technology': 0.4,
                                'Consumer Cyclical': 0.15,
                                'Communication Services': 0.15,
                                'Financial': 0.1,
                                'Healthcare': 0.1,
                                'Consumer Defensive': 0.1
                            }
                        }
                    }
                
                return {
                    'allocations': {},
                    'metrics': {
                        'predicted_return': 0,
                        'diversification_score': 0,
                        'sector_exposure': {}
                    }
                }
            
            # Score stocks
            stock_scores = {}
            stock_metrics = {}  # Store metrics for each stock
            
            for symbol in all_stocks:
                try:
                    # Get stock data
                    hist_data = self._get_historical_data(symbol)
                    if hist_data.empty:
                        continue
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(symbol, hist_data)
                    
                    # Get predicted return
                    predicted_return = self._predict_return(symbol)
                    if predicted_return != 0:
                        metrics['predicted_return'] = predicted_return
                    
                    # Store metrics
                    stock_metrics[symbol] = metrics
                    
                    # Calculate score
                    score = self._calculate_score(metrics)
                    stock_scores[symbol] = score
                    
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {str(e)}")
                    continue
            
            # Sort stocks by score
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Determine how many stocks to include based on risk profile
            if risk_profile == 'conservative':
                num_stocks = min(12, len(sorted_stocks))
            elif risk_profile == 'moderate':
                num_stocks = min(8, len(sorted_stocks))
            else:  # aggressive
                num_stocks = min(5, len(sorted_stocks))
            
            # Allocate investment
            total_score = sum(score for _, score in sorted_stocks[:num_stocks])
            allocations = {}
            
            for symbol, score in sorted_stocks[:num_stocks]:
                allocation_percent = score / total_score
                allocations[symbol] = amount * allocation_percent
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(allocations, stock_metrics)
            
            logger.info(f"Stock allocation complete with {len(allocations)} stocks")
            
            return {
                'allocations': allocations,
                'metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Error running stock allocation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return fallback allocations
            fallback_allocations = {
                'AAPL': amount * 0.3,  # Apple
                'MSFT': amount * 0.3,  # Microsoft
                'AMZN': amount * 0.2,  # Amazon
                'GOOGL': amount * 0.2  # Google
            }
            
            fallback_metrics = {
                'predicted_return': 0.07,  # 7% expected return
                'dividend_yield': 0.02,
                'beta': 1.1,
                'pe_ratio': 25,
                'diversification_score': 0.4,
                'sector_exposure': {'Technology': 0.8, 'Consumer Cyclical': 0.2}
            }
            
            return {
                'allocations': fallback_allocations,
                'metrics': fallback_metrics
            } 