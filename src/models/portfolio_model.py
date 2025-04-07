from models.mpt_allocation import MPTAllocationModel
from models.stock_allocation import StockAllocationModel
from models.crypto_allocation import CryptoAllocationModel
from models.gold_allocation import GoldAllocationModel
from models.bond_allocation import BondAllocationModel
from models.sentiment_model import SentimentModel
import logging
from typing import Dict
import traceback
import numpy as np

logger = logging.getLogger(__name__)

class PortfolioModel:
    """
    Main portfolio model that integrates all individual models and
    provides a complete investment recommendation.
    """
    
    def __init__(self):
        """Initialize the portfolio model with all component models."""
        self.mpt_model = MPTAllocationModel()
        self.stock_model = StockAllocationModel()
        self.crypto_model = CryptoAllocationModel()
        self.gold_model = GoldAllocationModel()
        self.bond_model = BondAllocationModel()
        self.sentiment_model = SentimentModel()
        
        # Cached results
        self.portfolio_data = None
    
    def _map_risk_profile(self, risk_tolerance: str) -> str:
        """
        Map risk tolerance to a risk profile category.
        
        Args:
            risk_tolerance (str): User's risk tolerance (Low, Moderate, High)
            
        Returns:
            str: Risk profile category
        """
        # Convert to lowercase for case-insensitive matching
        risk_tolerance_lower = risk_tolerance.lower() if isinstance(risk_tolerance, str) else ""
        
        # Map UI inputs to risk profiles
        risk_map = {
            # Standard inputs
            'low': 'conservative',
            'moderate': 'moderate',
            'high': 'aggressive',
            'very high': 'very_aggressive',
            
            # Alternative inputs from UI
            'very low': 'conservative',
            'medium': 'moderate',  # UI sometimes uses "medium" instead of "moderate"
            'medium-high': 'aggressive',
            'high risk': 'aggressive',
            'very aggressive': 'very_aggressive',
            
            # Numeric inputs (if passed as strings)
            '1': 'conservative',
            '2': 'moderate',
            '3': 'aggressive',
            '4': 'very_aggressive'
        }
        
        # Log the mapping
        logger.debug("Mapping risk tolerance '%s' to profile", risk_tolerance)
        profile = risk_map.get(risk_tolerance_lower, 'moderate')
        logger.debug("Mapped to profile: %s", profile)
        
        return profile
    
    def _calculate_risk_score(self, risk_profile: str) -> float:
        """
        Calculate numeric risk score based on risk profile.
        
        Args:
            risk_profile (str): Risk profile category
            
        Returns:
            float: Risk score between 0 and 1
        """
        risk_scores = {
            'conservative': 0.25,
            'moderately_conservative': 0.4,
            'moderate': 0.55,
            'aggressive': 0.75,
            'very_aggressive': 0.9
        }
        
        return risk_scores.get(risk_profile, 0.55)
    
    def _get_market_sentiment(self) -> Dict:
        """
        Get market sentiment for all asset classes.
        
        Returns:
            Dict: Sentiment analysis for each asset class
        """
        sentiment = {}
        try:
            for asset in ['stocks', 'bonds', 'crypto', 'gold']:
                sentiment[asset] = self.sentiment_model.get_asset_sentiment(asset)
            return sentiment
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            # Return default neutral sentiment
            return {
                'stocks': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}},
                'bonds': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}},
                'crypto': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}},
                'gold': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}}
            }
    
    def _adjust_allocation_by_sentiment(self, allocation: Dict, sentiment: Dict) -> Dict:
        """
        Adjust allocation based on sentiment analysis.
        
        Args:
            allocation (Dict): Base allocation from risk profile
            sentiment (Dict): Sentiment analysis for each asset class
            
        Returns:
            Dict: Adjusted allocation
        """
        try:
            adjusted = allocation.copy()
            sentiment_weight = 0.1  # 10% weight for sentiment
            
            # Calculate sentiment scores
            sentiment_scores = {}
            for asset, data in sentiment.items():
                sentiment_scores[asset] = data.get('overall_sentiment', 0)
            
            # Calculate average sentiment to determine if we should increase or decrease allocations
            avg_sentiment = sum(sentiment_scores.values()) / len(sentiment_scores) if sentiment_scores else 0
            
            # Adjust allocations based on relative sentiment
            for asset in adjusted:
                asset_sentiment = sentiment_scores.get(asset, 0)
                
                # Compare to average sentiment
                relative_sentiment = asset_sentiment - avg_sentiment
                
                # Apply adjustment (limited to +/- 5%)
                adjustment = max(-0.05, min(0.05, relative_sentiment * sentiment_weight))
                
                # Apply adjustment to percentage
                original_pct = adjusted[asset]['percentage']
                adjusted[asset]['percentage'] = max(0.05, min(0.7, original_pct + adjustment))
                
                # Recalculate amount
                adjusted[asset]['amount'] = adjusted[asset]['percentage'] * allocation[asset]['amount'] / original_pct
            
            # Normalize percentages to sum to 1
            total_pct = sum(asset['percentage'] for asset in adjusted.values())
            if total_pct != 0:
                for asset in adjusted:
                    adjusted[asset]['percentage'] /= total_pct
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error adjusting allocation by sentiment: {str(e)}")
            return allocation
    
    def generate_portfolio(self, user_input: Dict) -> Dict:
        """
        Generate a portfolio based on the user inputs.
        
        Args:
            user_input (Dict): User inputs from the web form.
            
        Returns:
            Dict: Portfolio recommendation.
        """
        try:
            logger.info("Generating portfolio with input: %s", str(user_input))
            
            # Extract user inputs
            salary = float(user_input.get('salary', 0))
            investment_goals = user_input.get('investment_goals', '')
            risk_tolerance = user_input.get('risk_tolerance', '')
            time_horizon = user_input.get('time_horizon', '')
            investment_amount = float(user_input.get('investment_amount', 0))
            
            logger.debug("Parsed inputs - salary: %s, goals: %s, risk: %s, horizon: %s, amount: %s", 
                         salary, investment_goals, risk_tolerance, time_horizon, investment_amount)
            
            # Validate numeric inputs
            if np.isnan(salary) or np.isnan(investment_amount):
                logger.error("Invalid numeric input: NaN detected in salary or investment amount")
                raise ValueError("Invalid numeric input: salary or investment amount is NaN")
            
            # Map risk profile based on inputs
            risk_profile = self._map_risk_profile(risk_tolerance)
            risk_score = self._calculate_risk_score(risk_profile)
            logger.debug("Mapped risk - profile: %s, score: %s", risk_profile, risk_score)
            
            # Create user profile
            user_profile = {
                'salary': salary,
                'investment_goals': investment_goals,
                'risk_tolerance': risk_tolerance,
                'time_horizon': time_horizon,
                'investment_amount': investment_amount,
                'risk_profile': risk_profile,
                'risk_score': risk_score
            }
            
            # Risk-based fallback allocations if needed
            risk_allocations = {
                'conservative': {
                    'stocks': 0.25,
                    'bonds': 0.55,
                    'crypto': 0.05,
                    'gold': 0.15
                },
                'moderate': {
                    'stocks': 0.40,
                    'bonds': 0.35, 
                    'crypto': 0.15,
                    'gold': 0.10
                },
                'aggressive': {
                    'stocks': 0.60,
                    'bonds': 0.15,
                    'crypto': 0.20,
                    'gold': 0.05
                },
                'very_aggressive': {
                    'stocks': 0.70,
                    'bonds': 0.05,
                    'crypto': 0.20,
                    'gold': 0.05
                }
            }
            
            logger.info("Step 1: Running Modern Portfolio Theory (MPT) optimization...")
            mpt_allocation = {}
            try:
                # Run MPT model 
                mpt_result = self.mpt_model.calculate_allocation(user_input)
                logger.debug("MPT result: %s", str(mpt_result))
                
                if isinstance(mpt_result, dict) and 'allocation' in mpt_result:
                    mpt_allocation = mpt_result['allocation']
                    logger.info("MPT allocation successful: %s", str(mpt_allocation))
                    
                    # Verify MPT allocation has valid values
                    if sum(mpt_allocation.values()) < 0.9 or sum(mpt_allocation.values()) > 1.1:
                        logger.warning("MPT allocation sum outside valid range (0.9-1.1): %s", sum(mpt_allocation.values()))
                        logger.warning("Using risk-based allocation instead")
                        mpt_allocation = risk_allocations.get(risk_profile, risk_allocations['moderate'])
                else:
                    logger.warning("MPT returned unexpected format: %s", str(mpt_result))
                    mpt_allocation = risk_allocations.get(risk_profile, risk_allocations['moderate'])
            except Exception as e:
                logger.error("Error in MPT calculation: %s", str(e))
                logger.error(traceback.format_exc())
                # Fallback to default allocation
                logger.warning("Using risk-based allocation due to MPT error")
                mpt_allocation = risk_allocations.get(risk_profile, risk_allocations['moderate'])
            
            logger.info("Step 2: Analyzing market sentiment for each asset class...")
            sentiment = {}
            try:
                sentiment = self._get_market_sentiment()
                logger.debug("Sentiment analysis: %s", str(sentiment))
            except Exception as e:
                logger.error("Error in sentiment analysis: %s", str(e))
                logger.error(traceback.format_exc())
                # Use neutral sentiment as fallback
                sentiment = {
                    'stocks': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}},
                    'bonds': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}},
                    'crypto': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}},
                    'gold': {'overall_sentiment': 0, 'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}}
                }
            
            logger.info("Step 3: Adjusting MPT allocation based on sentiment analysis...")
            adjusted_percentages = {}
            try:
                # Ensure MPT allocation is a simple dictionary of asset:percentage pairs
                logger.debug("MPT allocation before sentiment adjustment: %s", str(mpt_allocation))
                
                # Try sentiment adjustment
                adjusted_percentages = self.sentiment_model.adjust_allocation(mpt_allocation, sentiment)
                logger.debug("Allocation after sentiment adjustment: %s", str(adjusted_percentages))
                
                # Validate the adjusted allocation
                if not adjusted_percentages or sum(adjusted_percentages.values()) < 0.9 or sum(adjusted_percentages.values()) > 1.1:
                    logger.warning("Sentiment adjustment produced invalid allocation: %s", str(adjusted_percentages))
                    logger.warning("Using original MPT allocation instead")
                    adjusted_percentages = mpt_allocation
            except Exception as e:
                logger.error("Error adjusting allocation: %s", str(e))
                logger.error(traceback.format_exc())
                # Use MPT allocation as fallback
                adjusted_percentages = mpt_allocation
            
            # Convert percentages to amounts
            logger.info("Step 4: Converting percentages to amounts...")
            asset_allocation = {}
            for asset, percentage in adjusted_percentages.items():
                amount = percentage * investment_amount
                asset_allocation[asset] = {
                    'percentage': percentage,
                    'amount': amount
                }
                logger.debug("Asset %s: percentage = %.2f, amount = %.2f", asset, percentage, amount)
            
            # Double-check for any zero or very small allocations and replace with minimums
            for asset in ['stocks', 'bonds', 'crypto', 'gold']:
                if asset not in asset_allocation or asset_allocation.get(asset, {}).get('percentage', 0) < 0.05:
                    logger.warning(f"Asset {asset} has too small allocation, setting minimum values")
                    min_percent = 0.05
                    asset_allocation[asset] = {
                        'percentage': min_percent,
                        'amount': min_percent * investment_amount
                    }
            
            # Renormalize percentages if needed after adjustments
            total_percent = sum(data['percentage'] for data in asset_allocation.values())
            if abs(total_percent - 1.0) > 0.01:  # If not close to 100%
                logger.warning(f"Total allocation ({total_percent:.2f}) not close to 1.0, renormalizing")
                for asset in asset_allocation:
                    asset_allocation[asset]['percentage'] /= total_percent
                    asset_allocation[asset]['amount'] = asset_allocation[asset]['percentage'] * investment_amount
            
            logger.info("Step 5: Running individual asset class models for specific security selection...")
            specific_allocations = {}
            
            # Stocks allocation
            stock_amount = asset_allocation.get('stocks', {}).get('amount', 0)
            if stock_amount > 0:
                logger.info("Running stock allocation model (Random Forest)...")
                try:
                    specific_allocations['stocks'] = self.stock_model.run(stock_amount, risk_profile)
                    logger.debug("Stock allocation results - top 3: %s", 
                                 str(list(specific_allocations['stocks'].get('allocations', {}).items())[:3]))
                except Exception as e:
                    logger.error("Error in stock allocation: %s", str(e))
                    logger.error(traceback.format_exc())
                    specific_allocations['stocks'] = {'allocations': {'SPY': stock_amount * 0.7, 'VTI': stock_amount * 0.3}}
            
            # Bonds allocation
            bond_amount = asset_allocation.get('bonds', {}).get('amount', 0)
            if bond_amount > 0:
                logger.info("Running bond allocation model (LSTM Neural Network)...")
                try:
                    specific_allocations['bonds'] = self.bond_model.run(bond_amount, risk_profile)
                    logger.debug("Bond allocation results: rate env: %s, yield: %s", 
                                 specific_allocations['bonds'].get('rate_environment', 'unknown'),
                                 specific_allocations['bonds'].get('expected_yield', 0))
                except Exception as e:
                    logger.error("Error in bond allocation: %s", str(e))
                    logger.error(traceback.format_exc())
                    specific_allocations['bonds'] = {'allocations': {'BND': bond_amount * 0.8, 'AGG': bond_amount * 0.2}}
            
            # Crypto allocation
            crypto_amount = asset_allocation.get('crypto', {}).get('amount', 0)
            if crypto_amount > 0:
                logger.info("Running cryptocurrency allocation model (XGBoost)...")
                try:
                    specific_allocations['crypto'] = self.crypto_model.run(crypto_amount, risk_profile)
                    logger.debug("Crypto allocation results - top 2: %s", 
                                 str(list(specific_allocations['crypto'].get('allocations', {}).items())[:2]))
                except Exception as e:
                    logger.error("Error in crypto allocation: %s", str(e))
                    logger.error(traceback.format_exc())
                    specific_allocations['crypto'] = {'allocations': {'BTC': crypto_amount * 0.6, 'ETH': crypto_amount * 0.4}}
            
            # Gold allocation
            gold_amount = asset_allocation.get('gold', {}).get('amount', 0)
            if gold_amount > 0:
                logger.info("Running gold allocation model...")
                try:
                    specific_allocations['gold'] = self.gold_model.run(gold_amount, risk_profile)
                    logger.debug("Gold allocation results: %s", 
                                 specific_allocations['gold'].get('outlook', 'unknown'))
                except Exception as e:
                    logger.error("Error in gold allocation: %s", str(e))
                    logger.error(traceback.format_exc())
                    specific_allocations['gold'] = {'allocations': {'GLD': gold_amount}, 'outlook': 'neutral'}
            
            # Perform validation and handle NaN values
            logger.info("Validating allocations for NaN values...")
            for asset_class, allocations in specific_allocations.items():
                if allocations is None:
                    logger.warning(f"Received None for {asset_class} allocation")
                    continue
                    
                # Standard checking for all nested values
                for key, value in allocations.items():
                    if isinstance(value, (int, float)) and np.isnan(value):
                        logger.warning(f"NaN value found in {asset_class}.{key}, setting to 0")
                        allocations[key] = 0.0
                    # Check nested dictionaries for NaN values
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)) and np.isnan(subvalue):
                                logger.warning(f"NaN value found in {asset_class}.{key}.{subkey}, setting to 0")
                                value[subkey] = 0.0
                            # Specifically check for fields with '_change' in the name
                            if '_change' in subkey and isinstance(subvalue, (int, float)) and np.isnan(subvalue):
                                logger.warning(f"NaN value found in {asset_class}.{key}.{subkey}, setting to 0")
                                value[subkey] = 0.0
            
            logger.info("Step 6: Finalizing portfolio with all components...")
            # Store the generated portfolio
            self.portfolio_data = {
                'user_profile': user_profile,
                'asset_allocation': asset_allocation,
                'specific_allocations': specific_allocations,
                'sentiment_analysis': sentiment,
                'mpt_allocation': mpt_allocation
            }
            
            # Final NaN check - recursively check all nested objects
            self._validate_no_nan(self.portfolio_data)
            
            logger.info("Portfolio generation complete")
            return self.portfolio_data
            
        except Exception as e:
            logger.error(f"Error generating portfolio: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to generate portfolio: {str(e)}"}
    
    def _validate_no_nan(self, obj):
        """
        Recursively validate that there are no NaN values in the object.
        Replaces any found NaN values with 0.0.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    obj[key] = 0.0
                elif isinstance(value, (dict, list)):
                    self._validate_no_nan(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (int, float)) and np.isnan(item):
                    obj[i] = 0.0
                elif isinstance(item, (dict, list)):
                    self._validate_no_nan(item)
        return obj
    
    def get_summary(self):
        """
        Get a summary of the portfolio recommendation.
        
        Returns:
            dict: Summary of the portfolio recommendation.
        """
        if not self.portfolio_data:
            return {"error": "No portfolio has been generated yet."}
        
        try:
            # Extract key information for the summary
            user_profile = self.portfolio_data.get('user_profile', {})
            asset_allocation = self.portfolio_data.get('asset_allocation', {})
            specific_allocations = self.portfolio_data.get('specific_allocations', {})
            sentiment_analysis = self.portfolio_data.get('sentiment_analysis', {})
            
            # Create summary structure
            summary = {
                'risk_profile': user_profile.get('risk_profile', 'unknown'),
                'risk_score': user_profile.get('risk_score', 0),
                'asset_allocation': {
                    'stocks': asset_allocation.get('stocks', {}).get('percentage', 0),
                    'bonds': asset_allocation.get('bonds', {}).get('percentage', 0),
                    'crypto': asset_allocation.get('crypto', {}).get('percentage', 0),
                    'gold': asset_allocation.get('gold', {}).get('percentage', 0)
                },
                'top_stocks': [],
                'top_crypto': [],
                'bond_strategy': None,
                'gold_outlook': None,
                'predicted_returns': {},
                'sentiment_summary': {}
            }
            
            # Add sentiment data summary
            for asset_class, sentiment_data in sentiment_analysis.items():
                if 'overall_sentiment' in sentiment_data:
                    summary['sentiment_summary'][asset_class] = {
                        'sentiment': sentiment_data['overall_sentiment'],
                        'news_count': sentiment_data.get('news_count', 0)
                    }
            
            # Add top stock picks
            stock_data = specific_allocations.get('stocks', {})
            if 'allocations' in stock_data:
                # Sort stocks by amount and take top 5
                sorted_stocks = sorted(stock_data['allocations'].items(), key=lambda x: x[1], reverse=True)[:5]
                
                for ticker, amount in sorted_stocks:
                    stock_info = {
                        'ticker': ticker,
                        'amount': amount
                    }
                    
                    if 'names' in stock_data and ticker in stock_data['names']:
                        stock_info['name'] = stock_data['names'][ticker]
                    
                    summary['top_stocks'].append(stock_info)
                
                # Add predicted portfolio return
                if 'metrics' in stock_data and 'predicted_return' in stock_data['metrics']:
                    summary['predicted_returns']['stocks'] = stock_data['metrics']['predicted_return']
            
            # Add top crypto picks
            crypto_data = specific_allocations.get('crypto', {})
            if 'allocations' in crypto_data:
                # Sort crypto by amount and take top 3
                sorted_crypto = sorted(crypto_data['allocations'].items(), key=lambda x: x[1], reverse=True)[:3]
                
                for ticker, amount in sorted_crypto:
                    crypto_info = {
                        'ticker': ticker,
                        'amount': amount
                    }
                    
                    if 'names' in crypto_data and ticker in crypto_data['names']:
                        crypto_info['name'] = crypto_data['names'][ticker]
                    
                    summary['top_crypto'].append(crypto_info)
                
                # Add predicted return
                if 'metrics' in crypto_data and 'predicted_return' in crypto_data['metrics']:
                    summary['predicted_returns']['crypto'] = crypto_data['metrics']['predicted_return']
            
            # Add bond strategy info
            bond_data = specific_allocations.get('bonds', {})
            if bond_data:
                if 'rate_environment' in bond_data:
                    rate_env = bond_data['rate_environment']
                    strategy = "Focus on short-term bonds" if rate_env == 'rising_rates' else \
                             ("Balanced duration approach" if rate_env == 'stable_rates' else \
                              "Focus on long-term bonds")
                    
                    summary['bond_strategy'] = {
                        'environment': rate_env,
                        'strategy': strategy,
                        'expected_yield': bond_data.get('expected_yield', 0)
                    }
                
                # Add predicted return
                if 'predicted_portfolio_return' in bond_data:
                    summary['predicted_returns']['bonds'] = bond_data['predicted_portfolio_return']
            
            # Add gold outlook
            gold_data = specific_allocations.get('gold', {})
            if gold_data and 'outlook' in gold_data:
                summary['gold_outlook'] = gold_data['outlook']
                
                # Add predicted return
                if 'predicted_return' in gold_data:
                    summary['predicted_returns']['gold'] = gold_data['predicted_return']
            
            # Calculate overall predicted return
            if summary['predicted_returns']:
                weighted_returns = 0
                total_weight = 0
                
                for asset_class, predicted_return in summary['predicted_returns'].items():
                    weight = summary['asset_allocation'].get(asset_class, 0)
                    weighted_returns += predicted_return * weight
                    total_weight += weight
                
                summary['overall_predicted_return'] = weighted_returns / total_weight if total_weight > 0 else 0
            
            # Validate no NaN values in the summary
            self._validate_no_nan(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to generate summary: {str(e)}"} 