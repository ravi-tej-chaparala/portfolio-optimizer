import pandas as pd
import numpy as np
import requests
import yfinance as yf
from typing import Dict, List
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import random
import os
import json
from curl_cffi import requests as curl_requests

logger = logging.getLogger(__name__)

# Create a cache directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

class CryptoAllocationModel:
    """
    Cryptocurrency allocation model.
    Generates crypto recommendations based on machine learning and market data.
    """

    def __init__(self):
        """Initialize the crypto allocation model."""
        # Top cryptocurrencies to consider
        self.top_cryptos = [
            'BTC-USD',    # Bitcoin
            'ETH-USD',    # Ethereum
            'BNB-USD',    # Binance Coin
            'SOL-USD',    # Solana
            'ADA-USD',    # Cardano
            'XRP-USD',    # Ripple
            'DOT-USD',    # Polkadot
            'AVAX-USD',   # Avalanche
            'DOGE-USD',   # Dogecoin
            'MATIC-USD'   # Polygon
        ]
        
        # Smaller set for rate-limited situations
        self.core_cryptos = [
            'BTC-USD',    # Bitcoin
            'ETH-USD',    # Ethereum
            'BNB-USD',    # Binance Coin
        ]
        
        # Scoring weights
        self.score_weights = {
            'market_cap': 0.15,
            'volume': 0.10,
            'volatility': 0.15,
            'momentum': 0.15,
            'sharpe_ratio': 0.10,
            'predicted_return': 0.35  # ML prediction gets highest weight
        }
        
        # Base allocation weights for BTC and ETH based on risk profile
        self.base_allocations = {
            'conservative': {'BTC-USD': 0.6, 'ETH-USD': 0.3, 'others': 0.1},
            'moderate': {'BTC-USD': 0.5, 'ETH-USD': 0.3, 'others': 0.2},
            'aggressive': {'BTC-USD': 0.4, 'ETH-USD': 0.3, 'others': 0.3}
        }
        
        self.models = {}  # Cache for trained models
        self.cache = {}   # Cache for crypto data
        self.rate_limited = False
        self.req_count = 0
        self.last_req_time = pd.Timestamp.now()
    
        # Load cached data if available
        self._load_cache()
    
        # Create curl_cffi session
        self.session = curl_requests.Session(impersonate="chrome")
    
    def _respect_rate_limit(self):
        """
        Respect rate limits by adding delays between requests.
        """
        self.req_count += 1
        now = pd.Timestamp.now()
        
        # If we've made too many requests in a short time, slow down
        if self.req_count > 30:  # Reset counter after 30 requests (lower than stocks)
            self.req_count = 0
            time.sleep(3)  # 3 second pause after every 30 requests
        
        # Add a small delay between each request
        time_since_last = (now - self.last_req_time).total_seconds()
        if time_since_last < 0.3:  # Ensure at least 0.3 seconds between requests
            delay = 0.3 - time_since_last
            time.sleep(delay)
            
        self.last_req_time = pd.Timestamp.now()
    
    def _save_cache(self):
        """
        Save the cache to disk.
        """
        cache_file = os.path.join(CACHE_DIR, 'crypto_data_cache.json')
        try:
            serializable_cache = {}
            for symbol, data in self.cache.items():
                if isinstance(data, pd.DataFrame):
                    serializable_cache[symbol] = data.to_json()
                else:
                    serializable_cache[symbol] = data
                    
            with open(cache_file, 'w') as f:
                json.dump(serializable_cache, f)
                
            logger.info(f"Saved crypto data cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error saving crypto cache: {str(e)}")
    
    def _load_cache(self):
        """
        Load the cache from disk.
        """
        cache_file = os.path.join(CACHE_DIR, 'crypto_data_cache.json')
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
                        logger.warning(f"Error deserializing crypto cache for {symbol}: {str(e)}")
                
                logger.info(f"Loaded crypto data cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Error loading crypto cache: {str(e)}")
    
    def _get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical price data for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
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
            
            # Get data from Yahoo Finance
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
    
    def _calculate_metrics(self, symbol: str, hist_data: pd.DataFrame) -> Dict:
        """
        Calculate various metrics for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            hist_data (pd.DataFrame): Historical price data
            
        Returns:
            Dict: Dictionary of calculated metrics
        """
        try:
            # Calculate returns
            hist_data['daily_return'] = hist_data['Close'].pct_change()
            
            # Skip if not enough data
            if len(hist_data) < 30:
                logger.warning(f"Not enough data for {symbol}")
                return {
                    'volatility': 0,
                    'momentum': 0,
                    'sharpe_ratio': 0,
                    'market_cap': 0,
                    'volume': 0
                }
            
            # Calculate metrics
            volatility = hist_data['daily_return'].std() * np.sqrt(252)  # Annualize
            
            # Momentum (recent 30-day return)
            momentum = hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-30] - 1
            
            # Sharpe ratio (using 1% as risk-free rate)
            mean_return = hist_data['daily_return'].mean() * 252  # Annualize
            sharpe_ratio = (mean_return - 0.01) / volatility if volatility > 0 else 0
            
            # Market data
            latest_data = hist_data.iloc[-1]
            market_cap = latest_data.get('Close', 0) * latest_data.get('Volume', 0)  # Approximation
            volume = latest_data.get('Volume', 0)
            
            return {
                'volatility': volatility,
                'momentum': momentum,
                'sharpe_ratio': sharpe_ratio,
                'market_cap': market_cap,
                'volume': volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
            return {
                'volatility': 0,
                'momentum': 0,
                'sharpe_ratio': 0,
                'market_cap': 0,
                'volume': 0
            }
    
    def _predict_return(self, symbol: str, period: str = "1y") -> float:
        """
        Predict future returns for a cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol
            period (str): Period for historical data
            
        Returns:
            float: Predicted annual return
        """
        try:
            logger.info(f"Predicting return for {symbol}...")
            
            # Get historical data if not already cached
            hist_data = self._get_historical_data(symbol, period)
            
            if hist_data.empty or len(hist_data) < 60:  # Need at least 60 days of data
                logger.warning(f"Not enough data to predict returns for {symbol}, using default predictions")
                
                # Return default predictions based on symbol
                if 'BTC' in symbol:
                    default_return = 0.15  # 15% for Bitcoin
                    logger.info(f"Using default return for {symbol}: {default_return*100:.1f}%")
                    return default_return
                elif 'ETH' in symbol:
                    default_return = 0.18  # 18% for Ethereum
                    logger.info(f"Using default return for {symbol}: {default_return*100:.1f}%")
                    return default_return
                else:
                    default_return = 0.20  # 20% for other cryptocurrencies
                    logger.info(f"Using default return for {symbol}: {default_return*100:.1f}%")
                    return default_return
            
            # Check if we already have a trained model for this symbol
            if symbol in self.models:
                logger.info(f"Using existing model for {symbol}")
                model = self.models[symbol]
            else:
                logger.info(f"Training new model for {symbol}")
                # Prepare features
                df = hist_data.copy()
                
                # Create features (rolling statistics)
                df['price_5d_ma'] = df['Close'].rolling(window=5).mean()
                df['price_10d_ma'] = df['Close'].rolling(window=10).mean()
                df['price_20d_ma'] = df['Close'].rolling(window=20).mean()
                df['price_60d_ma'] = df['Close'].rolling(window=60).mean()
                
                df['volume_5d_ma'] = df['Volume'].rolling(window=5).mean()
                df['volume_10d_ma'] = df['Volume'].rolling(window=10).mean()
                
                df['volatility_10d'] = df['Close'].pct_change().rolling(window=10).std()
                df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
                
                # Target: 30-day future return
                df['future_return_30d'] = df['Close'].shift(-30) / df['Close'] - 1
                
                # Drop NaN values
                df = df.dropna()
                
                if len(df) < 30:  # Not enough data after creating features
                    logger.warning(f"Not enough data after feature creation for {symbol}, returning 0")
                    return 0.0
                
                # Features and target
                features = ['price_5d_ma', 'price_10d_ma', 'price_20d_ma', 'price_60d_ma',
                           'volume_5d_ma', 'volume_10d_ma', 'volatility_10d', 'volatility_20d']
                
                X = df[features]
                y = df['future_return_30d']
                
                # Simple scaling
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train a simple model
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                model.fit(X_scaled, y)
                
                # Save model for future use
                self.models[symbol] = model
                logger.info(f"Model training completed for {symbol}")
            
            # Use the model to predict future returns
            logger.info(f"Preparing prediction data for {symbol}")
            # Prepare new data for prediction
            latest_data = hist_data.iloc[-60:].copy()
            
            # Create the same features as in training
            latest_data['price_5d_ma'] = latest_data['Close'].rolling(window=5).mean()
            latest_data['price_10d_ma'] = latest_data['Close'].rolling(window=10).mean()
            latest_data['price_20d_ma'] = latest_data['Close'].rolling(window=20).mean()
            latest_data['price_60d_ma'] = latest_data['Close'].rolling(window=60).mean()
            
            latest_data['volume_5d_ma'] = latest_data['Volume'].rolling(window=5).mean()
            latest_data['volume_10d_ma'] = latest_data['Volume'].rolling(window=10).mean()
            
            latest_data['volatility_10d'] = latest_data['Close'].pct_change().rolling(window=10).std()
            latest_data['volatility_20d'] = latest_data['Close'].pct_change().rolling(window=20).std()
            
            # Get the most recent complete data point
            latest_complete = latest_data.dropna().iloc[-1:][['price_5d_ma', 'price_10d_ma', 'price_20d_ma', 'price_60d_ma',
                                                           'volume_5d_ma', 'volume_10d_ma', 'volatility_10d', 'volatility_20d']]
            
            if len(latest_complete) == 0:
                logger.warning(f"No complete latest data for {symbol}")
                return 0.0
            
            # Scale the data using the same approach as during training
            latest_scaled = StandardScaler().fit_transform(latest_complete)
            
            # Predict the 30-day return
            predicted_30d_return = model.predict(latest_scaled)[0]
            logger.info(f"30-day predicted return for {symbol}: {predicted_30d_return*100:.2f}%")
            
            # Convert 30-day return to annual return (approximately)
            annual_return = (1 + max(-0.5, min(1.0, predicted_30d_return))) ** (365/30) - 1
            
            # Cap the prediction to be reasonable
            capped_return = max(0.05, min(0.40, annual_return))  # Between 5% and 40%
            logger.info(f"Annual predicted return for {symbol}: {capped_return*100:.2f}%")
            
            return capped_return
            
        except Exception as e:
            logger.error(f"Error predicting returns for {symbol}: {str(e)}")
            return 0.0
    
    def _calculate_score(self, metrics: Dict) -> float:
        """
        Calculate a score for a cryptocurrency based on metrics.
        
        Args:
            metrics (Dict): Dictionary of metrics
            
        Returns:
            float: Score value between 0 and 1
        """
        try:
            # Initialize score
            score = 0.0
            
            # Calculate normalized scores for each metric
            if 'market_cap' in metrics and metrics['market_cap'] > 0:
                # Log scale for market cap (higher is better)
                market_cap_score = min(1.0, np.log10(metrics['market_cap']) / 12)
                score += market_cap_score * self.score_weights['market_cap']
            
            if 'volume' in metrics and metrics['volume'] > 0:
                # Log scale for volume (higher is better)
                volume_score = min(1.0, np.log10(metrics['volume']) / 10)
                score += volume_score * self.score_weights['volume']
                
            if 'volatility' in metrics and metrics['volatility'] > 0:
                # Inverse for volatility (lower is better for conservative investing)
                # Cap at reasonable values
                volatility = min(2.0, max(0.05, metrics['volatility']))
                volatility_score = 1.0 - (volatility / 2.0)
                score += volatility_score * self.score_weights['volatility']
            
            if 'momentum' in metrics:
                # Momentum (scale to 0-1 range, with 1 being best)
                momentum = metrics['momentum']
                momentum_score = min(1.0, max(0.0, (momentum + 0.5) / 1.5))
                score += momentum_score * self.score_weights['momentum']
                
            if 'sharpe_ratio' in metrics:
                # Sharpe ratio (scale to 0-1 range)
                sharpe = metrics['sharpe_ratio']
                sharpe_score = min(1.0, max(0.0, sharpe / 3.0))
                score += sharpe_score * self.score_weights['sharpe_ratio']
                
            if 'predicted_return' in metrics:
                # Predicted return (scale to 0-1 range)
                pred_return = metrics['predicted_return']
                return_score = min(1.0, max(0.0, pred_return / 0.4))  # 40% max expected return
                score += return_score * self.score_weights['predicted_return']
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            return 0.0
    
    def _calculate_portfolio_metrics(self, allocations: Dict) -> Dict:
        """
        Calculate metrics for the overall crypto portfolio.
        
        Args:
            allocations (Dict): Dictionary of crypto allocations
            
        Returns:
            Dict: Portfolio metrics
        """
        try:
            # If no allocations, return empty metrics
            if not allocations:
                return {
                    'predicted_return': 0,
                    'volatility': 0,
                    'btc_dominance': 0,
                    'diversification_score': 0,
                    'crypto_count': 0
                }
            
            # Calculate total allocation
            total_allocation = sum(allocations.values())
            
            # BTC dominance
            btc_dominance = allocations.get('BTC-USD', 0) / total_allocation if total_allocation > 0 else 0
            
            # Crypto count
            crypto_count = len(allocations)
            
            # Diversification score (higher is better, 1.0 is max)
            if crypto_count <= 1:
                diversification_score = 0.0
            else:
                # Based on how evenly distributed the allocations are
                weights = [amount / total_allocation for amount in allocations.values()]
                # Herfindahl-Hirschman Index (HHI) - measure of concentration
                hhi = sum([w**2 for w in weights])
                # Convert to diversification score (1 - HHI, normalized)
                # HHI ranges from 1/n (perfect diversification) to 1 (full concentration)
                # Normalize to 0-1 range
                min_hhi = 1 / crypto_count
                diversification_score = min(1.0, (1 - hhi) / (1 - min_hhi))
            
            # Calculate weighted average predicted return
            predicted_return = 0
            volatility = 0
            
            for symbol, amount in allocations.items():
                weight = amount / total_allocation if total_allocation > 0 else 0
                
                # Get predicted return for this crypto
                try:
                    if symbol == 'BTC-USD':
                        crypto_return = 0.15  # Default for Bitcoin
                        crypto_volatility = 0.50  # Default volatility
                    elif symbol == 'ETH-USD':
                        crypto_return = 0.18  # Default for Ethereum
                        crypto_volatility = 0.65  # Default volatility
                    else:
                        crypto_return = 0.20  # Default for other cryptos
                        crypto_volatility = 0.80  # Default volatility
                    
                    # Try to get actual predicted values if available
                    hist_data = self._get_historical_data(symbol)
                    if not hist_data.empty:
                        metrics = self._calculate_metrics(symbol, hist_data)
                        crypto_volatility = metrics.get('volatility', crypto_volatility)
                        
                        # Try to predict return
                        predicted_crypto_return = self._predict_return(symbol)
                        if predicted_crypto_return > 0:
                            crypto_return = predicted_crypto_return
                    
                    predicted_return += crypto_return * weight
                    volatility += crypto_volatility * weight
                    
                except Exception as e:
                    logger.error(f"Error getting metrics for {symbol}: {str(e)}")
                    # Use default values
                    predicted_return += 0.15 * weight
                    volatility += 0.60 * weight
            
            return {
                'predicted_return': predicted_return,
                'volatility': volatility,
                'btc_dominance': btc_dominance,
                'diversification_score': diversification_score,
                'crypto_count': crypto_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                'predicted_return': 0.12,  # Default 12% return
                'volatility': 0.50,  # Default 50% volatility
                'btc_dominance': 0.5,  # Default 50% BTC dominance
                'diversification_score': 0.5,  # Default medium diversification
                'crypto_count': len(allocations)
            }
    
    def run(self, amount: float, risk_profile: str) -> Dict:
        """
        Run the crypto allocation model.
        
        Args:
            amount (float): Total investment amount
            risk_profile (str): Risk profile (conservative, moderate, aggressive)
            
        Returns:
            Dict: Crypto allocation results
        """
        # If we're rate limited, use a smaller set of cryptos
        crypto_list = self.core_cryptos if self.rate_limited else self.top_cryptos
        
        try:
            # If we're rate limited and have no cache data, use fallback allocations
            if self.rate_limited and len(self.cache) < 3:
                logger.warning("Using fallback crypto allocations due to rate limiting")
                
                # Default allocations based on risk profile
                base_alloc = self.base_allocations[risk_profile]
                
                allocations = {
                    'BTC-USD': amount * base_alloc['BTC-USD'],
                    'ETH-USD': amount * base_alloc['ETH-USD']
                }
                
                # Add a small allocation to one more crypto based on risk
                if risk_profile == 'conservative':
                    allocations['BNB-USD'] = amount * base_alloc['others']
                elif risk_profile == 'moderate':
                    allocations['BNB-USD'] = amount * (base_alloc['others'] * 0.6)
                    allocations['SOL-USD'] = amount * (base_alloc['others'] * 0.4)
                else:  # aggressive
                    allocations['BNB-USD'] = amount * (base_alloc['others'] * 0.4)
                    allocations['SOL-USD'] = amount * (base_alloc['others'] * 0.3)
                    allocations['ADA-USD'] = amount * (base_alloc['others'] * 0.3)
                
                # Return with fallback metrics
                return {
                    'allocations': allocations,
                    'metrics': {
                        'predicted_return': 0.12,  # 12% expected return
                        'volatility': 0.30,
                        'btc_dominance': base_alloc['BTC-USD'],
                        'diversification_score': 0.6,
                        'crypto_count': len(allocations)
                    }
                }
            
            # Score cryptocurrencies
            crypto_scores = {}
            for symbol in crypto_list:
                try:
                    # Get historical data
                    hist_data = self._get_historical_data(symbol)
                    
                    if hist_data.empty:
                        continue
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(symbol, hist_data)
                    
                    # Get predicted return
                    predicted_return = self._predict_return(symbol)
                    if predicted_return != 0:
                        metrics['predicted_return'] = predicted_return
                    
                    # Calculate score
                    score = self._calculate_score(metrics)
                    
                    # Apply BTC and ETH preference
                    if symbol == 'BTC-USD':
                        score *= 1.2  # 20% bonus for Bitcoin
                    elif symbol == 'ETH-USD':
                        score *= 1.1  # 10% bonus for Ethereum
                    
                    crypto_scores[symbol] = score
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # If no cryptos scored, return empty result
            if not crypto_scores:
                logger.error("No valid cryptocurrencies could be scored")
                return {
                    'allocations': {},
                    'metrics': {
                        'predicted_return': 0,
                        'volatility': 0,
                        'btc_dominance': 0,
                        'diversification_score': 0,
                        'crypto_count': 0
                    }
                }
            
            # Sort cryptos by score
            sorted_cryptos = sorted(crypto_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get base allocations for this risk profile
            base_alloc = self.base_allocations[risk_profile]
            
            # Allocate investment with base allocations for BTC and ETH
            allocations = {}
            
            # Ensure BTC and ETH are included
            btc_amount = amount * base_alloc['BTC-USD']
            eth_amount = amount * base_alloc['ETH-USD']
            remaining_amount = amount * base_alloc['others']
            
            allocations['BTC-USD'] = btc_amount
            allocations['ETH-USD'] = eth_amount
            
            # Allocate remaining amount to other cryptocurrencies
            other_cryptos = [c for c, _ in sorted_cryptos if c not in ['BTC-USD', 'ETH-USD']]
            
            # Limit number of other cryptos based on risk profile
            if risk_profile == 'conservative':
                other_count = min(1, len(other_cryptos))
            elif risk_profile == 'moderate':
                other_count = min(3, len(other_cryptos))
            else:  # aggressive
                other_count = min(5, len(other_cryptos))
            
            selected_others = other_cryptos[:other_count]
            
            # Distribute remaining amount proportionally to scores
            if selected_others:
                other_scores = {c: crypto_scores[c] for c in selected_others}
                total_score = sum(other_scores.values())
                
                for crypto, score in other_scores.items():
                    allocations[crypto] = remaining_amount * (score / total_score)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(allocations)
            
            return {
                'allocations': allocations,
                'metrics': portfolio_metrics
            }
            
        except Exception as e:
            logger.error(f"Error running crypto allocation: {str(e)}")
            return {
                'allocations': {},
                'metrics': {
                    'predicted_return': 0,
                    'volatility': 0,
                    'btc_dominance': 0,
                    'diversification_score': 0,
                    'crypto_count': 0
                }
            } 