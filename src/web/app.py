from flask import Flask, request, jsonify, render_template
import sys
import os
import traceback
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
import numpy as np
import re
import openai

# Add the parent directory to the path to allow imports from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# Load environment variables
load_dotenv()
logger = setup_logging()


# JSON utilities
def sanitize_nan_values(obj):
    """
    Recursively replace all NaN values with None in a nested object.
    Works on dictionaries, lists, and primitive types.
    """
    if isinstance(obj, dict):
        return {k: sanitize_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan_values(item) for item in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN values."""

    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)


# Initialize services
def init_portfolio_model():
    """Initialize the portfolio model."""
    try:
        # Import using absolute imports
        from src.models.portfolio_model import PortfolioModel

        logger.info("Portfolio model initialized successfully")
        return PortfolioModel()
    except ImportError as e:
        logger.error(f"Failed to import PortfolioModel: {str(e)}")
        # Try alternative import paths
        try:
            import sys
            # Add project root to path if needed
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Try importing with a more direct path
            from models.portfolio_model import PortfolioModel
            logger.info("Portfolio model initialized with alternative import")
            return PortfolioModel()
        except ImportError as e2:
            logger.error(f"Failed to import with alternative method: {str(e2)}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize PortfolioModel: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def init_openai_client():
    """Initialize the OpenAI client."""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        logger.info(
            f"Loading OpenAI API key: {openai_api_key[:5] if openai_api_key else 'None'}..."
        )

        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            # Set the API key for the client
            openai.api_key = openai_api_key
            logger.info("OpenAI client initialized successfully")
            return True
        else:
            logger.warning("OpenAI API key not configured")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def init_data_service():
    """Initialize the financial data service."""
    try:
        # Import using absolute imports
        from src.utils.data_service import financial_data_service

        logger.info("Data service initialized successfully")
        return financial_data_service
    except ImportError as e:
        logger.error(f"Failed to import data service: {str(e)}")
        # Try alternative import paths
        try:
            import sys
            # Add project root to path if needed
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Try importing with a more direct path
            from utils.data_service import financial_data_service
            logger.info("Data service initialized with alternative import")
            return financial_data_service
        except ImportError as e2:
            logger.error(f"Failed to import with alternative method: {str(e2)}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize data service: {str(e)}")
        logger.error(traceback.format_exc())
        return None


# Initialize services
portfolio_model = init_portfolio_model()
openai_client = init_openai_client()
data_service = init_data_service()

# Initialize Flask app
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

app = Flask(
    __name__,
    static_folder=static_folder,
    static_url_path="/static",
    template_folder=template_folder,
)

# Set configuration from environment variables
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "default_secret_key")
app.config["API_TIMEOUT"] = int(os.getenv("API_TIMEOUT", 30))
app.config["MAX_PORTFOLIO_SIZE"] = int(os.getenv("MAX_PORTFOLIO_SIZE", 25))

# Set custom JSON encoder
app.json_encoder = CustomJSONEncoder


# Route handlers
@app.route("/")
def index():
    """Render the home page."""
    return render_template("index.html")


@app.route("/portfolio", methods=["POST"])
def generate_portfolio():
    """
    Generate a portfolio based on user input.

    Expected JSON format:
    {
        "salary": 100000,
        "investment_goals": "Retirement",
        "risk_tolerance": "Moderate",
        "time_horizon": "Long-term",
        "investment_amount": 10000
    }
    """
    try:
        # Check if portfolio model is initialized
        if not portfolio_model:
            logger.error("Portfolio model not initialized")
            return jsonify({"error": "Portfolio model not initialized"}), 500

        # Log incoming request
        logger.info(f"Received portfolio request: {request.json}")

        # Parse and validate request data
        data = request.json
        validation_result = validate_portfolio_request(data)

        if validation_result:
            logger.error(f"Validation error: {validation_result}")
            return jsonify({"error": validation_result}), 400

        # Generate portfolio
        logger.info("Starting portfolio generation with data: %s", data)
        portfolio = portfolio_model.generate_portfolio(data)

        # Debug log for development
        logger.info(
            f"Portfolio generation result (first 100 chars): {str(portfolio)[:100]}..."
        )

        # Validate portfolio result
        if portfolio is None:
            logger.error("Portfolio generation returned None")
            return jsonify(
                {"error": "Portfolio generation failed with null result"}
            ), 500

        if isinstance(portfolio, dict) and not portfolio:
            logger.error("Portfolio generation returned empty dictionary")
            return jsonify({"error": "Portfolio generation returned empty result"}), 500

        if isinstance(portfolio, dict) and "error" in portfolio:
            logger.error(f"Error generating portfolio: {portfolio['error']}")
            return jsonify(portfolio), 500

        # Validate portfolio structure
        required_sections = ["user_profile", "asset_allocation", "specific_allocations"]
        for section in required_sections:
            if section not in portfolio:
                logger.error(f"Portfolio missing required section: {section}")
                return jsonify(
                    {"error": f"Portfolio missing required section: {section}"}
                ), 500

        # Sanitize NaN values in the entire portfolio object
        sanitized_portfolio = sanitize_nan_values(portfolio)

        # Log success
        logger.info("Successfully generated portfolio")
        logger.debug(
            f"Number of assets in allocation: {len(sanitized_portfolio['asset_allocation'])}"
        )
        logger.debug(
            f"Asset classes: {list(sanitized_portfolio['asset_allocation'].keys())}"
        )

        return jsonify(sanitized_portfolio)

    except ValueError as e:
        logger.error(f"Invalid numerical value: {str(e)}")
        return jsonify({"error": f"Invalid numerical value: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error in portfolio generation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def validate_portfolio_request(data):
    """
    Validate the portfolio generation request data.

    Args:
        data: The request data to validate

    Returns:
        str: Error message if validation fails, None if validation succeeds
    """
    # Check if data is provided
    if not data:
        return "No request data provided"

    # Validate required fields
    required_fields = [
        "salary",
        "investment_goals",
        "risk_tolerance",
        "time_horizon",
        "investment_amount",
    ]

    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"

    # Convert salary and investment_amount to float
    try:
        data["salary"] = float(data["salary"])
        data["investment_amount"] = float(data["investment_amount"])
    except ValueError:
        return "Salary and investment amount must be valid numbers"

    return None


@app.route("/portfolio/summary", methods=["GET"])
def get_portfolio_summary():
    """Get a summary of the generated portfolio."""
    if not portfolio_model:
        return jsonify({"error": "Portfolio model not initialized"}), 500

    summary = portfolio_model.get_summary()
    # Sanitize NaN values in the summary
    sanitized_summary = sanitize_nan_values(summary)
    return jsonify(sanitized_summary)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat messages from the user and return responses using the LLM.
    Integrates portfolio data to answer questions about allocations.
    """
    try:
        # Validate request
        data = request.json
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify(
                {
                    "response": "Please enter a question about investing or your portfolio."
                }
            ), 400

        # Check if OpenAI client is available
        if not openai_client:
            logger.error("OpenAI client is not initialized")
            return jsonify(
                {
                    "response": "OpenAI is not configured. Please add your OpenAI API key to the .env file."
                }
            ), 400

        # Create prompt with portfolio data
        if portfolio_model:
            # Get the portfolio data
            portfolio_data = portfolio_model.get_summary()

            # Create the context with user message and portfolio data
            context = {
                "user_message": user_message,
                "portfolio_data": _format_portfolio_for_llm(portfolio_data),
                "instruction": "You are a helpful investment advisor assistant. Answer the user's question about their portfolio based on the provided portfolio data. Provide clear, accurate, and detailed explanations without using predefined responses.",
            }

            # Generate detailed response from LLM
            response = generate_chat_response(context)
        else:
            # If no portfolio data is available yet
            prompt = f"""You are a helpful investment advisor assistant. The user has asked: "{user_message}"
            
No portfolio has been generated yet, so you cannot provide specific portfolio details.
Explain that you need a portfolio to be generated first, or answer general investment questions.
"""
            response = generate_chat_response(prompt)

        return jsonify({"response": response})

    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify(
            {
                "response": "I'm having trouble answering right now. Please try again later."
            }
        ), 500


def _format_portfolio_for_llm(portfolio_data):
    """
    Format portfolio data in a structured way for the LLM.

    Args:
        portfolio_data (dict): The portfolio data

    Returns:
        str: Formatted portfolio data string
    """
    if (
        not portfolio_data
        or isinstance(portfolio_data, dict)
        and "error" in portfolio_data
    ):
        return "No portfolio data available."

    # Format risk profile
    user_profile = portfolio_data.get("user_profile", {})
    risk_profile = user_profile.get("risk_profile", "Unknown")
    risk_score = user_profile.get("risk_score", "Unknown")

    # Format asset allocation
    asset_allocation = portfolio_data.get("asset_allocation", {})
    allocation_text = ""
    for asset, data in asset_allocation.items():
        # Handle both dictionary format and direct numeric values
        if isinstance(data, dict):
            percentage = data.get("percentage", 0) * 100
            amount = data.get("amount", 0)
        else:
            # If it's a direct numeric value (numpy.float64 or other numeric type)
            percentage = float(data) * 100
            # Since we don't have amount, we'll calculate it based on total investment
            total_investment = portfolio_data.get("total_investment", 0)
            amount = total_investment * float(data)

        allocation_text += (
            f"- {asset.capitalize()}: {percentage:.1f}% (${amount:,.2f})\n"
        )

    # Format top picks
    top_stocks = portfolio_data.get("top_stocks", [])
    stock_text = ", ".join([f"{stock.get('ticker', '')}" for stock in top_stocks])

    top_crypto = portfolio_data.get("top_crypto", [])
    crypto_text = ", ".join([f"{crypto.get('ticker', '')}" for crypto in top_crypto])

    # Format predicted returns
    predicted_returns = portfolio_data.get("predicted_returns", {})
    overall_return = portfolio_data.get("overall_predicted_return", 0) * 100
    returns_text = f"Overall: {overall_return:.2f}%\n"
    for asset, return_pct in predicted_returns.items():
        returns_text += f"- {asset.capitalize()}: {return_pct * 100:.2f}%\n"

    # Format specific allocations
    specific_allocations = portfolio_data.get("specific_allocations", {})
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
        specific_text += f"- Market Outlook: {gold_data.get('outlook', 'neutral')}\n"

        gold_allocations = gold_data.get("allocations", {})
        specific_text += "Gold Allocations:\n"
        for symbol, amount in gold_allocations.items():
            specific_text += f"- {symbol}: ${amount:,.2f}\n"

    # Format reasoning for the allocation
    reasoning = portfolio_data.get("reasoning", "")
    reasoning_text = ""
    if reasoning:
        reasoning_text = f"\nREASONING:\n----------\n{reasoning}\n"

    # Model method information
    model_info = portfolio_data.get("model_info", {})
    model_text = ""
    if model_info:
        model_text = "\nMODEL INFORMATION:\n-----------------\n"
        for model_name, details in model_info.items():
            model_text += f"- {model_name}: {details}\n"

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
{reasoning_text}
{model_text}
"""

    return formatted_data


def build_chat_prompt(user_message):
    """
    Build the prompt for the chat model.

    Args:
        user_message: The user's message

    Returns:
        str: The prompt for the chat model
    """
    # Get portfolio data if it exists
    portfolio_data = {}
    if portfolio_model:
        portfolio_summary = portfolio_model.get_summary()
        # Only include portfolio data if a portfolio has been generated
        if isinstance(portfolio_summary, dict) and "error" not in portfolio_summary:
            portfolio_data = portfolio_summary

    # Prepare the prompt with portfolio data if available
    if portfolio_data:
        prompt = f"""You are a helpful investment advisor assistant. Provide clear, accurate, and helpful responses about investing, portfolio management, and financial planning.

Here is the user's current portfolio data:
```
Risk Profile: {portfolio_data.get("risk_profile", "N/A")}
Risk Score: {portfolio_data.get("risk_score", "N/A")}
Asset Allocation: 
- Stocks: {portfolio_data.get("asset_allocation", {}).get("stocks", 0) * 100:.1f}%
- Bonds: {portfolio_data.get("asset_allocation", {}).get("bonds", 0) * 100:.1f}%
- Crypto: {portfolio_data.get("asset_allocation", {}).get("crypto", 0) * 100:.1f}%
- Gold: {portfolio_data.get("asset_allocation", {}).get("gold", 0) * 100:.1f}%

Top Stock Picks: {", ".join([stock.get("ticker", "") for stock in portfolio_data.get("top_stocks", [])])}
Top Crypto Picks: {", ".join([crypto.get("ticker", "") for crypto in portfolio_data.get("top_crypto", [])])}

Predicted Returns:
- Overall: {portfolio_data.get("overall_predicted_return", 0) * 100:.2f}%
- Stocks: {portfolio_data.get("predicted_returns", {}).get("stocks", 0) * 100:.2f}%
- Bonds: {portfolio_data.get("predicted_returns", {}).get("bonds", 0) * 100:.2f}%
- Crypto: {portfolio_data.get("predicted_returns", {}).get("crypto", 0) * 100:.2f}%

Bond Strategy: {portfolio_data.get("specific_allocations", {}).get("bonds", {}).get("rate_environment", "N/A")}
```

The user asks: "{user_message}"

Respond to their question as a financial advisor would, using the portfolio data to provide a personalized response. Do not use generic templates - analyze their specific portfolio details to give a custom answer.
"""
    else:
        prompt = f"""You are a helpful investment advisor assistant. Provide clear, accurate, and helpful responses about investing, portfolio management, and financial planning.

The user has not generated a portfolio yet, so you don't have specific details about their investments.

The user asks: "{user_message}"

If they're asking about their specific portfolio, explain that they need to generate a portfolio first. Otherwise, provide general advice about their investment question.
"""

    return prompt


def generate_chat_response(prompt_or_context):
    """
    Generate a response using OpenAI.

    Args:
        prompt_or_context: Either a string prompt or a context object with
                           user_message, portfolio_data, and instruction fields

    Returns:
        str: The generated response
    """
    # Handle both string prompts and context objects
    if isinstance(prompt_or_context, dict):
        # Format the context object into a prompt
        context = prompt_or_context
        prompt = f"""You are a helpful investment advisor assistant. 
        
INSTRUCTION: {context.get("instruction", "Provide investment advice.")}

USER MESSAGE: {context.get("user_message", "")}

PORTFOLIO DATA:
{context.get("portfolio_data", "No portfolio data available.")}

CONVERSATION HISTORY:
{context.get("conversation_history", "")}

Provide a clear, accurate, and detailed response that directly addresses the user's question.
Explain portfolio concepts thoroughly based on the specific data provided.
Do not use template responses - analyze the provided information to give custom explanations.
"""
    else:
        # Use the string prompt directly
        prompt = prompt_or_context
    
    # Generate response with OpenAI
    if openai_client:
        logger.info("Generating response using OpenAI")
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",  # You can use gpt-4 if you have access
                messages=[
                    {"role": "system", "content": "You are a helpful investment advisor assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if not response_text:
                response_text = "I apologize, but I couldn't generate a proper response about your portfolio. Please try asking in a different way."
                
            logger.info(f"Generated response: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error during OpenAI API request: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "I'm having trouble analyzing your portfolio right now. Please try again later."
    
    # If no client is available
    else:
        logger.error("OpenAI client not available")
        return "I'm unable to analyze your portfolio as OpenAI is not configured. Please check API configuration."


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(debug=True, host="127.0.0.1", port=port)
