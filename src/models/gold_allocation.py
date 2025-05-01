import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import time
import random
import os
import json
from datetime import datetime, timedelta
from curl_cffi import requests

logger = logging.getLogger(__name__)


class GoldAllocationModel:
    """
    Gold allocation model that analyzes gold vs inflation trends.
    Determines optimal gold allocation based on current market conditions
    with rate limiting handling and caching to prevent API issues.
    """

    def __init__(self):
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "gold_cache.json")
        self.cache = self._load_cache()
        self.api_calls = 0
        self.max_api_calls = 25  # Maximum API calls before using fallback
        # Create curl_cffi session
        self.session = requests.Session(impersonate="chrome")

    def _respect_rate_limit(self):
        """Add delay between API calls and track call count to avoid rate limiting"""
        self.api_calls += 1
        if self.api_calls > self.max_api_calls:
            logger.warning("Rate limit approaching, using cache or fallback data")
            return False

        # Random delay between 1-3 seconds to avoid consistent patterns
        delay = random.uniform(1, 3)
        time.sleep(delay)
        return True

    def _load_cache(self):
        """Load cache from disk if it exists"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)

                # Check if cache is less than 30 days old
                if "timestamp" in cache_data:
                    cache_time = datetime.fromtimestamp(cache_data["timestamp"])
                    if datetime.now() - cache_time < timedelta(hours=720):
                        logger.info("Using gold data from cache")
                        return cache_data
                    else:
                        logger.info("Cache expired, will fetch fresh data")
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")

        return {"timestamp": datetime.now().timestamp(), "data": {}}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, "w") as f:
                self.cache["timestamp"] = datetime.now().timestamp()
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")

    def _get_inflation_data(self):
        """
        Get inflation rate using TIP vs IEF spread as a proxy
        with caching to prevent API overuse
        """
        try:
            cache_key = "inflation_proxy"

            # Check if we have cached data
            if cache_key in self.cache["data"]:
                logger.info("Using cached inflation proxy data")
                return self.cache["data"][cache_key]

            # Check if we should respect rate limit
            if not self._respect_rate_limit():
                return 0.03  # Default moderate inflation value

            # Get TIP (Treasury Inflation-Protected Securities) and IEF (7-10 Year Treasury) data
            tip = yf.Ticker("TIP", session=self.session)
            ief = yf.Ticker("IEF", session=self.session)

            # Get historical data for last 3 months
            tip_data = tip.history(period="3mo")
            ief_data = ief.history(period="3mo")

            if len(tip_data) == 0 or len(ief_data) == 0:
                logger.warning("Unable to fetch TIP or IEF data")
                return 0.03

            # Calculate spread and its recent change
            tip_price_change = (
                tip_data["Close"].iloc[-1] / tip_data["Close"].iloc[0]
            ) - 1
            ief_price_change = (
                ief_data["Close"].iloc[-1] / ief_data["Close"].iloc[0]
            ) - 1

            # If TIP outperforms IEF, inflation expectations are higher
            inflation_pressure = tip_price_change - ief_price_change

            # Map to roughly estimate inflation rate (approximation)
            estimated_inflation = 0.02 + (
                inflation_pressure * 5
            )  # Scale factor of 5 to amplify small differences
            estimated_inflation = max(
                0.0, min(0.1, estimated_inflation)
            )  # Cap between 0% and 10%

            # Cache the result
            self.cache["data"][cache_key] = estimated_inflation
            self._save_cache()

            return estimated_inflation
        except Exception as e:
            logger.error(f"Error getting inflation data: {e}")
            return 0.03  # Default to moderate inflation

    def _get_gold_to_sp500_ratio(self):
        """
        Get gold price trend relative to S&P 500
        with caching to prevent API overuse
        """
        try:
            cache_key = "gold_sp500_ratio"

            # Check if we have cached data
            if cache_key in self.cache["data"]:
                logger.info("Using cached gold/SP500 ratio data")
                return self.cache["data"][cache_key]

            # Check if we should respect rate limit
            if not self._respect_rate_limit():
                return {"current_ratio": 0.5, "ratio_change": 0.02}  # Default values

            # Get gold ETF (GLD) and S&P 500 ETF (SPY)
            gld = yf.Ticker("GLD", session=self.session)
            spy = yf.Ticker("SPY", session=self.session)

            # Get historical data for last 6 months
            gld_data = gld.history(period="6mo")
            spy_data = spy.history(period="6mo")

            if len(gld_data) == 0 or len(spy_data) == 0:
                logger.warning("Unable to fetch GLD or SPY data")
                return {"current_ratio": 0.5, "ratio_change": 0.02}

            # Calculate ratio at beginning and end of period
            start_ratio = gld_data["Close"].iloc[0] / spy_data["Close"].iloc[0]
            end_ratio = gld_data["Close"].iloc[-1] / spy_data["Close"].iloc[-1]

            # Calculate change in ratio
            ratio_change = (end_ratio / start_ratio) - 1

            result = {"current_ratio": end_ratio, "ratio_change": ratio_change}

            # Cache the result
            self.cache["data"][cache_key] = result
            self._save_cache()

            return result
        except Exception as e:
            logger.error(f"Error getting gold/SP500 ratio: {e}")
            return {"current_ratio": 0.5, "ratio_change": 0.02}

    def _get_vix_level(self):
        """
        Get market fear index (VIX) level
        with caching to prevent API overuse
        """
        try:
            cache_key = "vix"

            # Check if we have cached data
            if cache_key in self.cache["data"]:
                logger.info("Using cached VIX data")
                return self.cache["data"][cache_key]

            # Check if we should respect rate limit
            if not self._respect_rate_limit():
                return {
                    "current": 18,
                    "average": 20,
                    "state": "normal",
                }  # Default values

            # Get VIX data
            vix = yf.Ticker("^VIX", session=self.session)
            vix_data = vix.history(period="1mo")

            if len(vix_data) == 0:
                logger.warning("Unable to fetch VIX data")
                return {"current": 18, "average": 20, "state": "normal"}

            current_vix = vix_data["Close"].iloc[-1]
            avg_vix = vix_data["Close"].mean()

            # Determine market fear state
            if current_vix > 30:
                fear_state = "high"
            elif current_vix > 20:
                fear_state = "elevated"
            elif current_vix < 15:
                fear_state = "low"
            else:
                fear_state = "normal"

            result = {"current": current_vix, "average": avg_vix, "state": fear_state}

            # Cache the result
            self.cache["data"][cache_key] = result
            self._save_cache()

            return result
        except Exception as e:
            logger.error(f"Error getting VIX level: {e}")
            return {"current": 18, "average": 20, "state": "normal"}

    def _get_market_environment(self):
        """
        Determine overall market environment using cached or fresh data
        """
        try:
            # Get inflation data
            inflation = self._get_inflation_data()

            # Get gold to S&P 500 ratio
            gold_sp_ratio = self._get_gold_to_sp500_ratio()

            # Get VIX level (market fear)
            vix_data = self._get_vix_level()

            # Determine gold environment
            gold_favorable = False
            gold_env_description = "neutral"

            # High inflation is favorable for gold
            if inflation > 0.04:
                gold_favorable = True
                gold_env_description = "inflation_hedge"

            # Rising gold/SP500 ratio is favorable for gold
            if gold_sp_ratio["ratio_change"] > 0.05:
                gold_favorable = True
                gold_env_description = "outperforming_equities"

            # High market fear is favorable for gold
            if vix_data["state"] in ["elevated", "high"]:
                gold_favorable = True
                gold_env_description = "safe_haven"

            return {
                "inflation_rate": inflation,
                "gold_sp_ratio": gold_sp_ratio,
                "market_fear": vix_data,
                "gold_favorable": gold_favorable,
                "environment": gold_env_description,
            }
        except Exception as e:
            logger.error(f"Error evaluating market environment: {e}")
            return {
                "inflation_rate": 0.03,
                "gold_sp_ratio": {"current_ratio": 0.5, "ratio_change": 0.0},
                "market_fear": {"current": 20, "state": "normal"},
                "gold_favorable": False,
                "environment": "neutral",
            }

    def run(self, investment_amount, risk_profile=None):
        """
        Run the gold allocation model to get gold investment recommendations

        Args:
            investment_amount (float): Amount to invest in gold
            risk_profile (str, optional): Risk profile of the investor (conservative, moderate, aggressive)

        Returns:
            dict: Gold allocation recommendation with detailed metrics
        """
        try:
            logger.info(
                f"Running gold allocation model for ${investment_amount}, risk profile: {risk_profile}"
            )

            # Get current market environment
            market_env = self._get_market_environment()

            # Determine gold ETF allocation
            allocations = {}
            metrics = {}

            # Store metrics
            metrics = {
                "inflation_rate": market_env["inflation_rate"],
                "gold_sp_ratio": market_env["gold_sp_ratio"]["current_ratio"],
                "gold_sp_trend": market_env["gold_sp_ratio"]["ratio_change"],
                "market_fear": market_env["market_fear"]["current"],
                "fear_state": market_env["market_fear"]["state"],
                "environment": market_env["environment"],
            }

            # Adjust allocation based on risk profile if provided
            if risk_profile:
                risk_profile = risk_profile.lower()
                if risk_profile == "aggressive":
                    # More exposure to leveraged products for aggressive profiles
                    if market_env["gold_favorable"]:
                        allocations = {
                            "GLD": 0.6 * investment_amount,  # Standard gold ETF
                            "SGOL": 0.2 * investment_amount,  # Alternative gold ETF
                            "UGL": 0.2
                            * investment_amount,  # Higher leveraged gold exposure
                        }
                        outlook = "bullish"
                    else:
                        allocations = {
                            "GLD": 0.7 * investment_amount,  # Standard gold ETF
                            "SGOL": 0.2 * investment_amount,  # Alternative gold ETF
                            "UGL": 0.1
                            * investment_amount,  # Some leveraged gold even in neutral market
                        }
                        outlook = "neutral"
                elif risk_profile == "conservative":
                    # More conservative approach with no leverage
                    allocations = {
                        "GLD": 0.9 * investment_amount,  # Standard gold ETF
                        "SGOL": 0.1 * investment_amount,  # Alternative gold ETF
                    }
                    outlook = "neutral" if market_env["gold_favorable"] else "defensive"
                else:  # moderate or default
                    # Default balanced approach
                    if market_env["gold_favorable"]:
                        allocations = {
                            "GLD": 0.7 * investment_amount,  # Standard gold ETF
                            "SGOL": 0.2 * investment_amount,  # Alternative gold ETF
                            "UGL": 0.1
                            * investment_amount,  # Leveraged gold ETF for upside
                        }
                        outlook = "bullish"
                    else:
                        allocations = {
                            "GLD": 0.8 * investment_amount,  # Standard gold ETF
                            "SGOL": 0.2 * investment_amount,  # Alternative gold ETF
                        }
                        outlook = "neutral"
            else:
                # Original allocation logic if no risk profile is provided
                if market_env["gold_favorable"]:
                    allocations = {
                        "GLD": 0.7 * investment_amount,  # Standard gold ETF
                        "SGOL": 0.2 * investment_amount,  # Alternative gold ETF
                        "UGL": 0.1 * investment_amount,  # Leveraged gold ETF for upside
                    }
                    outlook = "bullish"
                else:
                    allocations = {
                        "GLD": 0.8 * investment_amount,  # Standard gold ETF
                        "SGOL": 0.2 * investment_amount,  # Alternative gold ETF
                    }
                    outlook = "neutral"

            return {
                "allocations": allocations,
                "total_amount": investment_amount,
                "metrics": metrics,
                "outlook": outlook,
                "method": "Rate-limited gold allocation with inflation & fear analysis",
            }

        except Exception as e:
            logger.error(f"Error in gold allocation model: {e}")
            # Fallback allocation in case of errors
            return {
                "allocations": {
                    "GLD": 0.8 * investment_amount,
                    "SGOL": 0.2 * investment_amount,
                },
                "total_amount": investment_amount,
                "metrics": {
                    "inflation_rate": 0.03,
                    "market_fear": "unknown",
                    "environment": "neutral",
                    "note": "Using fallback allocation due to data retrieval error",
                },
                "outlook": "neutral",
                "method": "Fallback allocation due to error",
            }
