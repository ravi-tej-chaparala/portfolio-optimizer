import re
import json
import os
from datetime import datetime


class PortfolioAssistant:
    """
    A chatbot implementation that relies on an LLM to explain portfolio recommendations and
    answer user questions about investments.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the portfolio assistant.

        Args:
            llm_client: Client for interacting with the LLM service
        """
        self.portfolio_data = None
        self.user_profile = None
        self.conversation_history = []
        self.llm_client = llm_client

    def set_portfolio_data(self, portfolio_data):
        """
        Set the portfolio data for the assistant to reference.

        Args:
            portfolio_data (dict): Portfolio recommendation data.
        """
        self.portfolio_data = portfolio_data
        if portfolio_data and "user_profile" in portfolio_data:
            self.user_profile = portfolio_data["user_profile"]

    def add_message_to_history(self, role, message):
        """
        Add a message to the conversation history.

        Args:
            role (str): 'user' or 'bot'
            message (str): The message content
        """
        self.conversation_history.append(
            {"role": role, "content": message, "timestamp": datetime.now().isoformat()}
        )

    def process_message(self, message, llm_client=None):
        """
        Process a user message and generate a response using an LLM.

        Args:
            message (str): User message.
            llm_client: Optional override for the LLM client

        Returns:
            str: Assistant response.
        """
        # Add user message to history
        self.add_message_to_history("user", message)

        # Use provided client or instance client
        client = llm_client or self.llm_client

        # Check if portfolio data is available
        if not self.portfolio_data:
            response = "I don't have access to your portfolio data yet. Please generate a portfolio first to get specific recommendations."
            self.add_message_to_history("bot", response)
            return response

        # Create a context for the LLM with portfolio data
        context = self._prepare_llm_context(message)

        # Get response from LLM
        try:
            if client:
                response = client.generate_response(context)
            else:
                # Fallback if no LLM client is available
                response = "I'm unable to process your request about the portfolio at this time. The LLM service is not available."
        except Exception as e:
            response = f"I encountered an error while generating a response: {str(e)}"

        # Add response to history
        self.add_message_to_history("bot", response)

        return response

    def _prepare_llm_context(self, user_message):
        """
        Prepare the context for the LLM with the user's message and portfolio data.

        Args:
            user_message (str): The user's message

        Returns:
            dict: Context for the LLM
        """
        # Format portfolio data for the LLM
        portfolio_summary = self._format_portfolio_data()

        # Format conversation history (last few messages)
        recent_history = (
            self.conversation_history[-5:] if len(self.conversation_history) > 1 else []
        )
        formatted_history = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]
        )

        # Create the context object
        context = {
            "user_message": user_message,
            "portfolio_data": portfolio_summary,
            "conversation_history": formatted_history,
            "instruction": "You are a helpful investment advisor assistant. Answer the user's question about their portfolio based on the provided portfolio data. Provide clear, accurate, and detailed explanations.",
        }

        return context

    def _format_portfolio_data(self):
        """
        Format the portfolio data for the LLM in a structured way.

        Returns:
            str: Formatted portfolio data
        """
        if not self.portfolio_data:
            return "No portfolio data available."

        # Extract key information
        portfolio = self.portfolio_data

        # Format risk profile info
        user_profile = portfolio.get("user_profile", {})
        risk_profile = user_profile.get("risk_profile", "Unknown")
        risk_score = user_profile.get("risk_score", "Unknown")

        # Format asset allocation
        asset_allocation = portfolio.get("asset_allocation", {})
        allocation_text = ""
        for asset, data in asset_allocation.items():
            percentage = data.get("percentage", 0) * 100
            amount = data.get("amount", 0)
            allocation_text += (
                f"- {asset.capitalize()}: {percentage:.1f}% (${amount:,.2f})\n"
            )

        # Format top picks
        top_stocks = portfolio.get("top_stocks", [])
        stock_text = ", ".join([f"{stock.get('ticker', '')}" for stock in top_stocks])

        top_crypto = portfolio.get("top_crypto", [])
        crypto_text = ", ".join(
            [f"{crypto.get('ticker', '')}" for crypto in top_crypto]
        )

        # Format predicted returns
        predicted_returns = portfolio.get("predicted_returns", {})
        overall_return = portfolio.get("overall_predicted_return", 0) * 100
        returns_text = f"Overall: {overall_return:.2f}%\n"
        for asset, return_pct in predicted_returns.items():
            returns_text += f"- {asset.capitalize()}: {return_pct * 100:.2f}%\n"

        # Format specific allocations
        specific_allocations = portfolio.get("specific_allocations", {})
        specific_text = ""

        # Stocks
        if "stocks" in specific_allocations:
            stock_allocations = specific_allocations["stocks"].get("allocations", {})
            specific_text += "Stock Allocations:\n"
            for symbol, amount in stock_allocations.items():
                specific_text += f"- {symbol}: ${amount:,.2f}\n"

        # Bonds
        if "bonds" in specific_allocations:
            bond_data = specific_allocations["bonds"]
            specific_text += "\nBond Strategy:\n"
            specific_text += (
                f"- Rate Environment: {bond_data.get('rate_environment', 'unknown')}\n"
            )
            specific_text += (
                f"- Expected Yield: {bond_data.get('expected_yield', 0):.2f}%\n"
            )

            bond_allocations = bond_data.get("allocations", {})
            specific_text += "Bond Allocations:\n"
            for symbol, amount in bond_allocations.items():
                specific_text += f"- {symbol}: ${amount:,.2f}\n"

        # Crypto
        if "crypto" in specific_allocations:
            crypto_allocations = specific_allocations["crypto"].get("allocations", {})
            specific_text += "\nCrypto Allocations:\n"
            for symbol, amount in crypto_allocations.items():
                specific_text += f"- {symbol}: ${amount:,.2f}\n"

        # Gold
        if "gold" in specific_allocations:
            gold_data = specific_allocations["gold"]
            specific_text += "\nGold Strategy:\n"
            specific_text += (
                f"- Market Outlook: {gold_data.get('outlook', 'neutral')}\n"
            )

            gold_allocations = gold_data.get("allocations", {})
            specific_text += "Gold Allocations:\n"
            for symbol, amount in gold_allocations.items():
                specific_text += f"- {symbol}: ${amount:,.2f}\n"

        # Combine all data into a formatted string
        formatted_data = f"""
PORTFOLIO SUMMARY:
------------------
Risk Profile: {risk_profile}
Risk Score: {risk_score}

ASSET ALLOCATION:
----------------
{allocation_text}

TOP PICKS:
---------
Stocks: {stock_text}
Crypto: {crypto_text}

PREDICTED RETURNS:
----------------
{returns_text}

SPECIFIC ALLOCATIONS:
-------------------
{specific_text}
"""

        return formatted_data
