import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import logging
import traceback
from curl_cffi import requests

logger = logging.getLogger(__name__)

class MPTAllocationModel:
    """
    Modern Portfolio Theory (MPT) based allocation model.
    Uses real market data to optimize portfolio allocation.
    """
    
    def __init__(self):
        """Initialize the MPT model."""
        self.assets = {
            'stocks': '^GSPC',  # S&P 500
            'bonds': '^TNX',    # 10-Year Treasury
            'crypto': 'BTC-USD', # Bitcoin
            'gold': 'GC=F'      # Gold Futures
        }
        # Initialize with a session for yfinance
        self.session = requests.Session(impersonate="chrome")
    
    def _get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate from 3-month Treasury bill.
        
        Returns:
            float: Current risk-free rate
        """
        try:
            tbill = yf.Ticker('^IRX', session=self.session)  # 13-week Treasury bill
            hist = tbill.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100  # Convert to decimal
            return 0.05  # Fallback to 5% if data unavailable
        except Exception as e:
            logger.error(f"Error getting risk-free rate: {str(e)}")
            return 0.05  # Fallback to 5% if error occurs
    
    def _get_historical_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Get historical data for all assets.
        
        Args:
            period (str): Time period for historical data
            
        Returns:
            pd.DataFrame: Historical returns data
        """
        try:
            data = {}
            for asset, symbol in self.assets.items():
                ticker = yf.Ticker(symbol, session=self.session)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[asset] = hist['Close'].pct_change().dropna()
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns: pd.DataFrame) -> tuple:
        """
        Calculate portfolio return and risk.
        
        Args:
            weights (np.ndarray): Asset weights
            returns (pd.DataFrame): Historical returns
            
        Returns:
            tuple: (portfolio_return, portfolio_risk)
        """
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return portfolio_return, portfolio_risk
    
    def _objective_function(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """
        Objective function for portfolio optimization (Sharpe Ratio).
        
        Args:
            weights (np.ndarray): Asset weights
            returns (pd.DataFrame): Historical returns
            
        Returns:
            float: Negative Sharpe ratio (for minimization)
        """
        portfolio_return, portfolio_risk = self._calculate_portfolio_metrics(weights, returns)
        risk_free_rate = self._get_risk_free_rate()
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        return -sharpe_ratio
    
    def _optimize_portfolio(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Optimize portfolio weights using MPT.
        
        Args:
            returns (pd.DataFrame): Historical returns
            
        Returns:
            np.ndarray: Optimal weights
        """
        num_assets = len(self.assets)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # weights >= 0
        )
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_weights = np.array([1/num_assets] * num_assets)
        
        try:
            result = minimize(
                self._objective_function,
                initial_weights,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            return result.x
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return initial_weights
    
    def calculate_allocation(self, user_input: dict) -> dict:
        """
        Calculate optimal portfolio allocation based on user input.
        
        Args:
            user_input (dict): User preferences and constraints
            
        Returns:
            dict: Portfolio allocation and risk score
        """
        try:
            logger.info("MPT model running with user input: %s", str(user_input))
            
            # Get historical data
            returns = self._get_historical_data()
            if returns.empty:
                logger.warning("No historical data available, using default equal allocation")
                return {
                    'allocation': {asset: 0.25 for asset in self.assets},
                    'score': 0.5
                }
            
            # Analyze the data we received
            logger.debug("Historical returns shape: %s", str(returns.shape))
            logger.debug("Returns stats: min=%s, max=%s, mean=%s", 
                         returns.min().min(), returns.max().max(), returns.mean().mean())
            
            # Optimize portfolio
            logger.info("Running MPT optimization...")
            optimal_weights = self._optimize_portfolio(returns)
            logger.debug("Optimal weights before adjustment: %s", str(optimal_weights))
            
            # Calculate risk score based on portfolio risk
            _, portfolio_risk = self._calculate_portfolio_metrics(optimal_weights, returns)
            risk_score = min(1.0, portfolio_risk / 0.3)  # Normalize risk score
            
            # Get risk tolerance from user input (handle various formats)
            risk_tolerance = user_input.get('risk_tolerance', 'Moderate')
            if isinstance(risk_tolerance, str):
                risk_tolerance = risk_tolerance.lower()
            
            # Map risk tolerance to adjustment factor
            risk_tolerance_map = {
                'low': 0.7,
                'very low': 0.6,
                'medium-low': 0.8,
                'medium': 1.0,
                'moderate': 1.0,
                'medium-high': 1.2,
                'high': 1.3,
                'very high': 1.5
            }
            
            risk_factor = risk_tolerance_map.get(risk_tolerance, 1.0)
            logger.debug("Risk adjustment factor: %s for tolerance: %s", risk_factor, risk_tolerance)
            
            # Adjust weights based on risk tolerance - increase/decrease stock/crypto exposure
            stock_index = list(self.assets.keys()).index('stocks')
            crypto_index = list(self.assets.keys()).index('crypto')
            bond_index = list(self.assets.keys()).index('bonds')
            gold_index = list(self.assets.keys()).index('gold')
            
            # Create a copy of weights to adjust
            adjusted_weights = optimal_weights.copy()
            
            # Higher risk: increase stocks/crypto, decrease bonds
            if risk_factor > 1.0:
                # Increase stocks and crypto proportionally to risk factor
                adjusted_weights[stock_index] *= risk_factor
                adjusted_weights[crypto_index] *= risk_factor
                # Decrease bonds inversely
                adjusted_weights[bond_index] /= risk_factor
            # Lower risk: decrease stocks/crypto, increase bonds
            elif risk_factor < 1.0:
                # Decrease stocks and crypto proportionally
                adjusted_weights[stock_index] *= risk_factor
                adjusted_weights[crypto_index] *= risk_factor
                # Increase bonds and gold inversely
                adjusted_weights[bond_index] /= risk_factor
                adjusted_weights[gold_index] /= risk_factor * 0.8  # Less adjustment for gold
            
            # Make sure no allocation goes below 5% or above 70%
            for i in range(len(adjusted_weights)):
                adjusted_weights[i] = max(0.05, min(0.7, adjusted_weights[i]))
            
            # Normalize weights to sum to 1
            adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
            
            logger.debug("Adjusted weights: %s", str(adjusted_weights))
            
            # Create allocation dictionary
            allocation = {
                asset: float(weight)
                for asset, weight in zip(self.assets.keys(), adjusted_weights)
            }
            
            logger.info("MPT allocation complete: %s", str(allocation))
            
            return {
                'allocation': allocation,
                'score': risk_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating allocation: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Use different default allocations based on risk tolerance
            risk_tolerance = str(user_input.get('risk_tolerance', 'moderate')).lower()
            
            if 'low' in risk_tolerance:
                default_allocation = {'stocks': 0.3, 'bonds': 0.5, 'crypto': 0.05, 'gold': 0.15}
            elif 'high' in risk_tolerance:
                default_allocation = {'stocks': 0.6, 'bonds': 0.15, 'crypto': 0.2, 'gold': 0.05}
            else: # moderate
                default_allocation = {'stocks': 0.4, 'bonds': 0.35, 'crypto': 0.15, 'gold': 0.1}
                
            logger.warning("Using default allocation due to error: %s", str(default_allocation))
            
            return {
                'allocation': default_allocation,
                'score': 0.5
            } 