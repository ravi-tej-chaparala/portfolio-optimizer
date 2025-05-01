from transformers import pipeline
import logging
from typing import Dict, List, Union, Optional
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import pandas as pd
import traceback
from curl_cffi import requests

logger = logging.getLogger(__name__)

class SentimentModel:
    """
    Sentiment analysis model for financial markets using Hugging Face transformers.
    Analyzes sentiment for different asset classes using real market data and news.
    """
    
    def __init__(self):
        """Initialize the sentiment analysis model."""
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # Use CPU
            )
            logger.info("Sentiment analysis model initialized successfully")
            self.indices = {
                "S&P 500": "^GSPC",
                "Nasdaq": "^IXIC", 
                "Dow Jones": "^DJI",
                "VIX": "^VIX",
                "Russell 2000": "^RUT"
            }
            self.lookback_periods = [7, 14, 30, 90]  # days
            # Create curl_cffi session
            self.session = requests.Session(impersonate="chrome")
        except Exception as e:
            logger.error(f"Error initializing sentiment analysis model: {str(e)}")
            raise
    
    def _get_market_data(self, symbol: str, period: str = "1mo") -> Dict[str, float]:
        """
        Get market data for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period for analysis
            
        Returns:
            Dict[str, float]: Market metrics
        """
        try:
            ticker = yf.Ticker(symbol, session=self.session)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {}
            
            # Calculate basic metrics
            returns = hist['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
            
            return {
                'volatility': volatility,
                'momentum': momentum,
                'volume_change': (hist['Volume'].iloc[-1] / hist['Volume'].iloc[0] - 1)
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}
    
    def _get_news_sentiment(self, query: str, days: int = 7) -> List[Dict[str, float]]:
        """
        Get news articles and their sentiment for a given query.
        
        Args:
            query (str): Search query
            days (int): Number of days to look back
            
        Returns:
            List[Dict[str, float]]: List of sentiment scores for news articles
        """
        try:
            # In a real implementation, you would use a news API service
            # This is a placeholder for demonstration
            return []
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            return []
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores (positive, negative, neutral)
        """
        try:
            result = self.sentiment_analyzer(text)[0]
            
            # Convert label to score
            label = result['label']
            score = result['score']
            
            # Initialize scores
            scores = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            }
            
            # Map the result to our score format
            if label == 'positive':
                scores['positive'] = score
                scores['negative'] = 0.0
                scores['neutral'] = 1 - score
            elif label == 'negative':
                scores['positive'] = 0.0
                scores['negative'] = score
                scores['neutral'] = 1 - score
            else:
                scores['positive'] = 0.0
                scores['negative'] = 0.0
                scores['neutral'] = 1.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
    
    def get_asset_sentiment(self, asset_type: str) -> Dict[str, float]:
        """
        Get sentiment analysis for a specific asset type using real market data.
        
        Args:
            asset_type (str): Type of asset (stocks, crypto, bonds, gold)
            
        Returns:
            Dict[str, float]: Sentiment scores and overall sentiment
        """
        try:
            # Define market indicators for each asset type
            indicators = {
                'stocks': {
                    'symbol': '^GSPC',  # S&P 500
                    'name': 'US Stock Market'
                },
                'crypto': {
                    'symbol': 'BTC-USD',  # Bitcoin
                    'name': 'Cryptocurrency Market'
                },
                'bonds': {
                    'symbol': '^TNX',  # 10-Year Treasury Yield
                    'name': 'Bond Market'
                },
                'gold': {
                    'symbol': 'GC=F',  # Gold Futures
                    'name': 'Gold Market'
                }
            }
            
            if asset_type not in indicators:
                logger.warning(f"No indicators found for asset type: {asset_type}")
                return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            
            # Get market data
            market_data = self._get_market_data(indicators[asset_type]['symbol'])
            
            # Get news sentiment
            news_sentiment = self._get_news_sentiment(indicators[asset_type]['name'])
            
            # Calculate sentiment based on market data
            market_sentiment = 0.0
            if market_data:
                # Combine multiple factors
                momentum_factor = np.tanh(market_data['momentum'] * 2)
                volatility_factor = 1 - np.tanh(market_data['volatility'] * 2)
                volume_factor = np.tanh(market_data['volume_change'])
                
                market_sentiment = (momentum_factor + volatility_factor + volume_factor) / 3
            
            # Combine market sentiment with news sentiment
            if news_sentiment:
                news_scores = [s['overall_sentiment'] for s in news_sentiment]
                news_avg = np.mean(news_scores)
                overall_sentiment = (market_sentiment + news_avg) / 2
            else:
                overall_sentiment = market_sentiment
            
            # Convert to sentiment scores
            if overall_sentiment > 0:
                scores = {
                    'positive': overall_sentiment,
                    'negative': 0.0,
                    'neutral': 1 - overall_sentiment
                }
            elif overall_sentiment < 0:
                scores = {
                    'positive': 0.0,
                    'negative': -overall_sentiment,
                    'neutral': 1 + overall_sentiment
                }
            else:
                scores = {
                    'positive': 0.33,
                    'negative': 0.33,
                    'neutral': 0.34
                }
            
            return {
                'scores': scores,
                'overall_sentiment': overall_sentiment,
                'market_data': market_data
            }
            
        except Exception as e:
            logger.error(f"Error getting asset sentiment: {str(e)}")
            return {
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'overall_sentiment': 0.0,
                'market_data': {}
            }
    
    def adjust_allocation(self, base_allocation: Dict[str, float], sentiment_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust allocation based on sentiment scores and market data.
        
        Args:
            base_allocation (Dict[str, float]): Base allocation from MPT
            sentiment_scores (Dict[str, float]): Sentiment scores for each asset
            
        Returns:
            Dict[str, float]: Adjusted allocation
        """
        try:
            # Log inputs for debugging
            logger.debug("Base allocation: %s", str(base_allocation))
            logger.debug("Sentiment scores: %s", str(sentiment_scores))
            
            # Safety check: if base allocation is empty, return default allocation
            if not base_allocation:
                logger.warning("Empty base allocation provided")
                return {'stocks': 0.40, 'bonds': 0.35, 'crypto': 0.15, 'gold': 0.10}
            
            # Create a copy of the base allocation
            adjusted = base_allocation.copy()
            
            # Calculate sentiment adjustment factor (small factor for conservative adjustments)
            sentiment_factor = 0.05  # 5% weight for sentiment
            
            # Convert any non-float values to floats
            for asset in adjusted:
                if not isinstance(adjusted[asset], (int, float)):
                    logger.warning(f"Non-numeric allocation for {asset}: {adjusted[asset]}, setting to 0.25")
                    adjusted[asset] = 0.25
            
            # Adjust each asset's allocation based on sentiment
            for asset in adjusted:
                if asset in sentiment_scores:
                    sentiment_data = sentiment_scores[asset]
                    sentiment = sentiment_data.get('overall_sentiment', 0)
                    
                    # Only apply sentiment if it's a number
                    if isinstance(sentiment, (int, float)):
                        # Cap the sentiment to avoid extreme adjustments
                        sentiment = max(-1.0, min(1.0, sentiment))
                        
                        # Adjust allocation (multiply by a factor between 0.95 and 1.05)
                        adjustment = 1.0 + (sentiment * sentiment_factor)
                        logger.debug(f"Asset {asset}: sentiment={sentiment}, adjustment={adjustment}")
                        adjusted[asset] = adjusted[asset] * adjustment
            
            # Normalize allocations to sum to 1
            total = sum(adjusted.values())
            
            # Safety check to prevent division by zero
            if total <= 0.001:  # Practically zero
                logger.warning("Total adjusted allocation is near zero, returning default allocation")
                return {'stocks': 0.40, 'bonds': 0.35, 'crypto': 0.15, 'gold': 0.10}
            
            # Normalize
            normalized = {k: v/total for k, v in adjusted.items()}
            logger.debug("Final adjusted allocation: %s", str(normalized))
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error adjusting allocation: {str(e)}")
            logger.error(traceback.format_exc())
            # Return moderate allocation as failsafe
            return {'stocks': 0.40, 'bonds': 0.35, 'crypto': 0.15, 'gold': 0.10} 