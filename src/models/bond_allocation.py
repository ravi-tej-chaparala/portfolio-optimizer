import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from curl_cffi import requests

logger = logging.getLogger(__name__)


class BondAllocationModel:
    """
    Bond allocation model.
    Generates bond recommendations based on machine learning and yield curve analysis.
    """

    def __init__(self):
        """Initialize the bond allocation model."""
        # Bond ETF mapping by type and duration
        self.bond_etfs = {
            "treasury": {
                "short": ["SHY", "VGSH"],  # 1-3 year Treasury
                "intermediate": ["IEF", "VGIT"],  # 3-10 year Treasury
                "long": ["TLT", "VGLT"],  # 10+ year Treasury
            },
            "corporate": {
                "short": ["VCSH", "SPSB"],  # Short-term corporate
                "intermediate": ["VCIT", "SPIB"],  # Intermediate-term corporate
                "long": ["VCLT", "SPLB"],  # Long-term corporate
            },
            "municipal": {
                "short": ["SHM", "VTEB"],  # Short-term muni
                "intermediate": ["MUB", "ITM"],  # Intermediate-term muni
                "long": ["TFI", "MLN"],  # Long-term muni
            },
            "high_yield": {
                "short": ["SHYG", "SJNK"],  # Short-term high yield
                "intermediate": ["HYG", "JNK"],  # Intermediate high yield
                "long": ["FALN", "HYLD"],  # Long-term high yield
            },
            "international": {
                "all": ["BNDX", "IAGG", "IGOV"]  # International bonds
            },
        }

        # Risk profile mappings (normalized to sum to 1)
        self.risk_profiles = {
            "conservative": {
                "treasury": 0.55,
                "municipal": 0.25,
                "corporate": 0.15,
                "international": 0.05,
                "high_yield": 0.0,
            },
            "moderate": {
                "treasury": 0.35,
                "municipal": 0.25,
                "corporate": 0.25,
                "international": 0.10,
                "high_yield": 0.05,
            },
            "aggressive": {
                "treasury": 0.15,
                "municipal": 0.15,
                "corporate": 0.40,
                "international": 0.15,
                "high_yield": 0.15,
            },
        }

        # Duration allocation by interest rate environment
        self.duration_allocations = {
            "rising_rates": {"short": 0.70, "intermediate": 0.25, "long": 0.05},
            "stable_rates": {"short": 0.30, "intermediate": 0.50, "long": 0.20},
            "falling_rates": {"short": 0.10, "intermediate": 0.40, "long": 0.50},
        }

        # Scoring weights
        self.score_weights = {
            "yield": 0.25,
            "expense_ratio": 0.15,
            "aum": 0.10,
            "predicted_return": 0.35,  # ML prediction gets highest weight
            "volatility": 0.15,
        }

        self.models = {}  # Cache for trained models

        # Create curl_cffi session
        self.session = requests.Session(impersonate="chrome")

    def _get_interest_rate_environment(self) -> str:
        """
        Determine the current interest rate environment based on yield curve.

        Returns:
            str: Rate environment (rising_rates, stable_rates, or falling_rates)
        """
        try:
            # Get 3-month Treasury rate
            try:
                short_rate = yf.Ticker("^IRX", session=self.session)
                short_hist = short_rate.history(
                    period="1mo"
                )  # Changed from "6m" to "1mo"
                short_rate_value = (
                    short_hist["Close"].iloc[-1] if not short_hist.empty else None
                )
            except Exception as e:
                logger.warning(f"Error getting short-term rate data: {str(e)}")
                short_rate_value = None

            # Get 10-year Treasury rate
            try:
                long_rate = yf.Ticker("^TNX", session=self.session)
                long_hist = long_rate.history(
                    period="1mo"
                )  # Changed from "6m" to "1mo"
                long_rate_value = (
                    long_hist["Close"].iloc[-1] if not long_hist.empty else None
                )
            except Exception as e:
                logger.warning(f"Error getting long-term rate data: {str(e)}")
                long_rate_value = None

            # Alternative fallback: use ETFs as proxies for rate trends
            if short_rate_value is None or long_rate_value is None:
                logger.info("Using ETFs as proxies for interest rate trends")
                try:
                    # SHY (1-3 year Treasury ETF) price movement as short-term rate proxy
                    shy = yf.Ticker("SHY", session=self.session)
                    shy_hist = shy.history(period="3mo")

                    # TLT (20+ year Treasury ETF) price movement as long-term rate proxy
                    tlt = yf.Ticker("TLT", session=self.session)
                    tlt_hist = tlt.history(period="3mo")

                    if not shy_hist.empty and not tlt_hist.empty:
                        # Bond prices move inversely to yields
                        short_term_change = -(
                            shy_hist["Close"].iloc[-1] / shy_hist["Close"].iloc[0] - 1
                        )
                        long_term_change = -(
                            tlt_hist["Close"].iloc[-1] / tlt_hist["Close"].iloc[0] - 1
                        )

                        # Determine environment based on short-term changes
                        if (
                            short_term_change > 0.02
                        ):  # 2% price decrease = rate increase
                            return "rising_rates"
                        elif (
                            short_term_change < -0.02
                        ):  # 2% price increase = rate decrease
                            return "falling_rates"
                        else:
                            return "stable_rates"
                    else:
                        logger.warning(
                            "Could not get proxy ETF data, defaulting to stable rates"
                        )
                        return "stable_rates"
                except Exception as e:
                    logger.warning(f"Error using ETF proxies: {str(e)}")
                    return "stable_rates"

            # If we have direct Treasury yield data, use it
            if short_hist.empty or long_hist.empty:
                logger.warning("Unable to get Treasury yield data")
                return "stable_rates"  # Default to stable

            # Get first available values within the time period
            first_short_idx = short_hist.index[0]
            first_long_idx = long_hist.index[0]

            # Calculate rate of change over available period
            short_rate_start = short_hist["Close"].iloc[0]
            short_rate_end = short_hist["Close"].iloc[-1]
            short_rate_change = short_rate_end - short_rate_start

            logger.info(f"Short rate change: {short_rate_change:.2f} basis points")

            # Determine environment based on short-term rate changes
            if short_rate_change > 0.25:  # 25 basis points (lowered threshold)
                return "rising_rates"
            elif short_rate_change < -0.25:  # -25 basis points (lowered threshold)
                return "falling_rates"
            else:
                return "stable_rates"

        except Exception as e:
            logger.error(f"Error determining interest rate environment: {str(e)}")
            return "stable_rates"  # Default to stable

    def _get_historical_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Get historical price data for a bond ETF.

        Args:
            ticker (str): ETF ticker symbol
            period (str): Time period for data

        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            etf = yf.Ticker(ticker, session=self.session)
            hist = etf.history(period=period)

            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return pd.DataFrame()

            return hist
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _prepare_features(
        self, data: pd.DataFrame, rate_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for machine learning model, including interest rate data.

        Args:
            data (pd.DataFrame): Historical price data
            rate_data (pd.DataFrame): Interest rate data

        Returns:
            pd.DataFrame: Feature dataframe
        """
        if data.empty:
            return pd.DataFrame()

        try:
            # Create a copy to avoid modifying the original
            df = data.copy()

            # Basic price features
            df["Returns"] = df["Close"].pct_change()
            df["MA5"] = df["Close"].rolling(window=5).mean()
            df["MA10"] = df["Close"].rolling(window=10).mean()
            df["MA20"] = df["Close"].rolling(window=20).mean()
            df["MA50"] = df["Close"].rolling(window=50).mean()

            # Ratio features
            df["MA5_Ratio"] = df["Close"] / df["MA5"]
            df["MA10_Ratio"] = df["Close"] / df["MA10"]
            df["MA20_Ratio"] = df["Close"] / df["MA20"]
            df["MA50_Ratio"] = df["Close"] / df["MA50"]

            # Volatility features
            df["Volatility5"] = df["Returns"].rolling(window=5).std()
            df["Volatility20"] = df["Returns"].rolling(window=20).std()
            df["Volatility50"] = df["Returns"].rolling(window=50).std()

            # Add interest rate data if available
            if not rate_data.empty:
                # Merge rate data with price data based on date
                rate_data = rate_data.reindex(df.index, method="ffill")
                if "Close" in rate_data.columns:
                    df["Rate"] = rate_data["Close"]
                    df["Rate_Change_5d"] = df["Rate"].pct_change(periods=5)
                    df["Rate_Change_20d"] = df["Rate"].pct_change(periods=20)

            # Price momentum features
            df["Momentum5"] = df["Close"].pct_change(periods=5)
            df["Momentum10"] = df["Close"].pct_change(periods=10)
            df["Momentum20"] = df["Close"].pct_change(periods=20)

            # Target variable - 20-day future return
            df["Target"] = df["Close"].pct_change(periods=20).shift(-20)

            # Drop NaN values
            df.dropna(inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _train_model(self, features: pd.DataFrame, ticker: str) -> tuple:
        """
        Train machine learning model to predict future returns.

        Args:
            features (pd.DataFrame): Feature dataframe
            ticker (str): ETF ticker symbol

        Returns:
            tuple: (model, scaler, feature_names, model_quality)
        """
        if features.empty:
            return None, None, None, 0

        try:
            # Define features and target
            feature_names = [col for col in features.columns if col != "Target"]
            feature_names = [
                "Returns",
                "MA5_Ratio",
                "MA20_Ratio",
                "MA50_Ratio",
                "Volatility5",
                "Volatility20",
                "Momentum5",
                "Momentum20",
            ]

            # Add rate features if available
            if "Rate" in features.columns:
                feature_names.extend(["Rate", "Rate_Change_5d", "Rate_Change_20d"])

            X = features[feature_names]
            y = features["Target"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

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

            logger.info(
                f"Trained model for {ticker} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}"
            )

            return model, scaler, feature_names, model_quality

        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            return None, None, None, 0

    def _predict_return(self, ticker: str) -> float:
        """
        Predict future return for a bond ETF using machine learning.

        Args:
            ticker (str): ETF ticker symbol

        Returns:
            float: Predicted return
        """
        try:
            # Get historical data
            hist_data = self._get_historical_data(ticker)
            if hist_data.empty:
                return 0.0

            # Get interest rate data
            rate_data = pd.DataFrame()
            try:
                # Get appropriate Treasury rate data based on bond type
                if ticker in ["SHY", "VGSH", "SPSB", "VCSH", "SHM", "SHYG", "SJNK"]:
                    # Short-term bonds - use 3-month rate
                    rate = yf.Ticker("^IRX", session=self.session)
                    rate_data = rate.history(period="2y")
                elif ticker in [
                    "IEF",
                    "VGIT",
                    "VCIT",
                    "SPIB",
                    "MUB",
                    "ITM",
                    "HYG",
                    "JNK",
                ]:
                    # Intermediate-term bonds - use 5-year rate
                    rate = yf.Ticker("^FVX", session=self.session)
                    rate_data = rate.history(period="2y")
                else:
                    # Long-term bonds - use 10-year rate
                    rate = yf.Ticker("^TNX", session=self.session)
                    rate_data = rate.history(period="2y")
            except Exception as e:
                logger.warning(f"Error getting rate data for {ticker}: {str(e)}")

            # Prepare features
            features = self._prepare_features(hist_data, rate_data)
            if features.empty:
                return 0.0

            # Train model if not cached
            if ticker not in self.models:
                model, scaler, feature_names, model_quality = self._train_model(
                    features, ticker
                )
                if model is None:
                    return 0.0
                self.models[ticker] = {
                    "model": model,
                    "scaler": scaler,
                    "feature_names": feature_names,
                    "quality": model_quality,
                }

            # Get current features
            current_features = features.iloc[-1:][self.models[ticker]["feature_names"]]

            # Scale and predict
            scaled_features = self.models[ticker]["scaler"].transform(current_features)
            predicted_return = self.models[ticker]["model"].predict(scaled_features)[0]

            logger.info(f"Predicted return for {ticker}: {predicted_return:.4f}")

            return predicted_return

        except Exception as e:
            logger.error(f"Error predicting return for {ticker}: {str(e)}")
            return 0.0

    def _get_bond_etf_data(self, ticker: str) -> Dict:
        """
        Get data for a bond ETF.

        Args:
            ticker (str): ETF ticker symbol

        Returns:
            Dict: ETF data and metrics
        """
        try:
            etf = yf.Ticker(ticker, session=self.session)
            hist = etf.history(period="1y")
            info = etf.info

            if hist.empty:
                return {}

            # Calculate metrics
            returns = hist["Close"].pct_change().dropna()
            ytd_return = hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0

            # Get yield information
            sec_yield = info.get("yield", 0)
            if sec_yield > 1:  # Convert from percentage to decimal if needed
                sec_yield /= 100

            # Predict future return using ML
            predicted_return = self._predict_return(ticker)

            return {
                "current_price": hist["Close"].iloc[-1],
                "ytd_return": ytd_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "yield": sec_yield,
                "expense_ratio": info.get("expenseRatio", 0),
                "nav": info.get("navPrice", hist["Close"].iloc[-1]),
                "aum": info.get("totalAssets", 0),
                "predicted_return": predicted_return,
            }
        except Exception as e:
            logger.error(f"Error getting data for {ticker}: {str(e)}")
            return {}

    def _select_etfs_by_type_and_duration(
        self, bond_type: str, duration: str
    ) -> Dict[str, float]:
        """
        Select and score ETFs for a specific bond type and duration.

        Args:
            bond_type (str): Type of bond (treasury, corporate, etc.)
            duration (str): Duration bucket (short, intermediate, long)

        Returns:
            Dict[str, float]: Dict of ETF tickers and their scores
        """
        try:
            if bond_type == "international":
                duration = "all"  # International bonds use 'all' category

            # Get ETFs for this type and duration
            etfs = self.bond_etfs.get(bond_type, {}).get(duration, [])
            if not etfs:
                return {}

            # Get data and score each ETF
            etf_scores = {}
            for ticker in etfs:
                data = self._get_bond_etf_data(ticker)
                if data:
                    # Calculate score components
                    yield_score = data.get("yield", 0) * 5  # Weight yield higher
                    expense_score = (
                        1 - data.get("expense_ratio", 0.01) * 20
                    )  # Lower expense is better
                    aum_score = min(
                        1.0, data.get("aum", 0) / 10000000000
                    )  # Normalize AUM
                    predicted_return_score = max(
                        0, data.get("predicted_return", 0) * 10
                    )  # Scale predicted return
                    volatility_score = 1 / (
                        1 + data.get("volatility", 0.05) * 10
                    )  # Lower volatility is better

                    # Calculate weighted score
                    total_score = (
                        yield_score * self.score_weights["yield"]
                        + expense_score * self.score_weights["expense_ratio"]
                        + aum_score * self.score_weights["aum"]
                        + predicted_return_score
                        * self.score_weights["predicted_return"]
                        + volatility_score * self.score_weights["volatility"]
                    )

                    # Adjust score based on model quality
                    model_quality = self.models.get(ticker, {}).get("quality", 0)
                    adjusted_score = total_score * (0.8 + 0.2 * model_quality)

                    etf_scores[ticker] = {"score": adjusted_score, "data": data}

            return etf_scores

        except Exception as e:
            logger.error(f"Error selecting ETFs for {bond_type} {duration}: {str(e)}")
            return {}

    def run(self, amount: float, risk_profile: str = "moderate") -> Dict:
        """
        Generate a bond allocation based on the user inputs.

        Args:
            amount (float): Amount to invest in bonds.
            risk_profile (str): User's risk profile (conservative, moderate, aggressive).

        Returns:
            Dict: Bond allocation recommendations.
        """
        try:
            if amount <= 0:
                return {"allocations": {}, "total": 0}

            # Normalize risk profile
            risk_profile = risk_profile.lower()
            if risk_profile not in self.risk_profiles:
                risk_profile = "moderate"

            # Get interest rate environment
            rate_environment = self._get_interest_rate_environment()
            logger.info(f"Current interest rate environment: {rate_environment}")

            # Get bond type allocation based on risk profile
            type_allocation = self.risk_profiles[risk_profile]

            # Get duration allocation based on interest rate environment
            duration_allocation = self.duration_allocations[rate_environment]

            # Allocate funds to each bond type and duration
            allocations = {}
            simplified_allocations = {}  # For frontend compatibility
            names = {}  # Store ETF names
            predicted_returns = []

            for bond_type, type_percent in type_allocation.items():
                if type_percent <= 0:
                    continue

                # Calculate amount for this bond type
                type_amount = amount * type_percent

                # Skip if amount is too small
                if type_amount < 100:
                    continue

                # For international bonds, we don't use duration buckets
                if bond_type == "international":
                    etf_scores = self._select_etfs_by_type_and_duration(
                        bond_type, "all"
                    )
                    if etf_scores:
                        # Select top ETF by score
                        top_etf = max(etf_scores.items(), key=lambda x: x[1]["score"])

                        if bond_type not in allocations:
                            allocations[bond_type] = {
                                "percentage": type_percent,
                                "amount": type_amount,
                                "tickers": {},
                            }

                        ticker = top_etf[0]
                        predicted_return = top_etf[1]["data"].get("predicted_return", 0)
                        predicted_returns.append(predicted_return)

                        # Ensure amount and yield are numeric values
                        ticker_amount = float(type_amount)
                        ticker_yield = float(top_etf[1]["data"].get("yield", 0))

                        allocations[bond_type]["tickers"][ticker] = {
                            "amount": ticker_amount,
                            "yield": ticker_yield,
                            "predicted_return": predicted_return,
                        }

                        # Add to simplified allocations
                        simplified_key = f"{ticker}"
                        simplified_allocations[simplified_key] = ticker_amount
                        names[simplified_key] = f"{bond_type} {ticker}"
                    continue

                # Allocate to different durations
                type_allocations = {}
                for duration, duration_percent in duration_allocation.items():
                    # Calculate amount for this duration
                    duration_amount = type_amount * duration_percent

                    # Skip if amount is too small
                    if duration_amount < 100:
                        continue

                    # Select ETF for this type and duration
                    etf_scores = self._select_etfs_by_type_and_duration(
                        bond_type, duration
                    )
                    if etf_scores:
                        # Select top ETF by score
                        top_etf = max(etf_scores.items(), key=lambda x: x[1]["score"])

                        if bond_type not in type_allocations:
                            type_allocations[bond_type] = {
                                "percentage": type_percent,
                                "amount": type_amount,
                                "tickers": {},
                            }

                        ticker = top_etf[0]
                        predicted_return = top_etf[1]["data"].get("predicted_return", 0)
                        predicted_returns.append(predicted_return)

                        # Ensure amount and yield are numeric values
                        ticker_amount = float(duration_amount)
                        ticker_yield = float(top_etf[1]["data"].get("yield", 0))

                        type_allocations[bond_type]["tickers"][ticker] = {
                            "amount": ticker_amount,
                            "yield": ticker_yield,
                            "predicted_return": predicted_return,
                        }

                        # Add to simplified allocations
                        simplified_key = f"{ticker}"
                        simplified_allocations[simplified_key] = ticker_amount
                        names[simplified_key] = f"{bond_type} {ticker}"

                # Add to overall allocations
                for bond_type, allocation in type_allocations.items():
                    if bond_type in allocations:
                        # Merge ticker allocations
                        for ticker, data in allocation["tickers"].items():
                            allocations[bond_type]["tickers"][ticker] = data
                    else:
                        allocations[bond_type] = allocation

            # Calculate expected yield and predicted return
            total_investment = 0
            weighted_yield = 0

            for bond_type, allocation in allocations.items():
                for ticker, data in allocation["tickers"].items():
                    amount = data["amount"]
                    ticker_yield = data["yield"]
                    total_investment += amount
                    weighted_yield += amount * ticker_yield

            expected_yield = (
                weighted_yield / total_investment if total_investment > 0 else 0
            )
            avg_predicted_return = (
                np.mean(predicted_returns) if predicted_returns else 0
            )

            # Check that simplified allocations has data
            if not simplified_allocations:
                # No ETFs were selected; add some default ones
                if amount >= 1000:
                    simplified_allocations = {
                        "BND": amount * 0.6,
                        "AGG": amount * 0.25,
                        "BNDX": amount * 0.15,
                    }
                    names = {
                        "BND": "Total Bond Market ETF",
                        "AGG": "US Aggregate Bond ETF",
                        "BNDX": "International Bond ETF",
                    }

            return {
                "allocations": simplified_allocations,  # Use simplified structure for frontend
                "nested_allocations": allocations,  # Keep detailed structure for backend
                "names": names,
                "total": amount,
                "risk_profile": risk_profile,
                "rate_environment": rate_environment,
                "expected_yield": expected_yield,
                "predicted_portfolio_return": avg_predicted_return,
                "method": "ML-based ETF selection with duration strategy",
            }

        except Exception as e:
            logger.error(f"Error running bond allocation: {str(e)}")
            # Return fallback allocation
            simplified_allocations = {
                "BND": amount * 0.6,
                "AGG": amount * 0.25,
                "BNDX": amount * 0.15,
            }
            names = {
                "BND": "Total Bond Market ETF",
                "AGG": "US Aggregate Bond ETF",
                "BNDX": "International Bond ETF",
            }

            return {
                "allocations": simplified_allocations,
                "names": names,
                "total": amount,
                "risk_profile": risk_profile,
                "rate_environment": "unknown",
                "expected_yield": 0.03,  # Default 3% yield
                "predicted_portfolio_return": 0.04,  # Default 4% return
                "method": "Default allocation (error recovery)",
            }
