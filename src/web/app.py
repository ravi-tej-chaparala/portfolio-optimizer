from flask import Flask, request, jsonify, render_template
import sys
import os
import traceback
import logging
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import json
import numpy as np

# Add the parent directory to the path to allow imports from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'app_{datetime.now().strftime("%Y%m%d")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
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
        from models.portfolio_model import PortfolioModel
        logger.info("Portfolio model initialized successfully")
        return PortfolioModel()
    except ImportError as e:
        logger.error(f"Failed to import PortfolioModel: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize PortfolioModel: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def init_hf_client():
    """Initialize the Hugging Face client."""
    try:
        hf_token = os.getenv('HUGGINGFACE_API_KEY')
        logger.info(f"Loading Hugging Face API key: {hf_token[:5] if hf_token else 'None'}...")
        
        if hf_token and hf_token != 'your_huggingface_api_key_here':
            client = InferenceClient(token=hf_token)
            logger.info("Hugging Face client initialized successfully")
            return client
        else:
            logger.warning("Hugging Face API key not configured")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face client: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def init_data_service():
    """Initialize the financial data service."""
    try:
        from utils.data_service import financial_data_service
        logger.info("Data service initialized successfully")
        return financial_data_service
    except Exception as e:
        logger.error(f"Failed to initialize data service: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Initialize services
portfolio_model = init_portfolio_model()
hf_client = init_hf_client()
data_service = init_data_service()

# Initialize Flask app
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, 
            static_folder=static_folder,
            static_url_path='/static',
            template_folder=template_folder)

# Set configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['API_TIMEOUT'] = int(os.getenv('API_TIMEOUT', 30))
app.config['MAX_PORTFOLIO_SIZE'] = int(os.getenv('MAX_PORTFOLIO_SIZE', 25))

# Set custom JSON encoder
app.json_encoder = CustomJSONEncoder

# Route handlers
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/portfolio', methods=['POST'])
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
        logger.info(f"Portfolio generation result (first 100 chars): {str(portfolio)[:100]}...")
        
        # Validate portfolio result
        if portfolio is None:
            logger.error("Portfolio generation returned None")
            return jsonify({"error": "Portfolio generation failed with null result"}), 500
        
        if isinstance(portfolio, dict) and not portfolio:
            logger.error("Portfolio generation returned empty dictionary")
            return jsonify({"error": "Portfolio generation returned empty result"}), 500
        
        if isinstance(portfolio, dict) and 'error' in portfolio:
            logger.error(f"Error generating portfolio: {portfolio['error']}")
            return jsonify(portfolio), 500
        
        # Validate portfolio structure
        required_sections = ['user_profile', 'asset_allocation', 'specific_allocations']
        for section in required_sections:
            if section not in portfolio:
                logger.error(f"Portfolio missing required section: {section}")
                return jsonify({"error": f"Portfolio missing required section: {section}"}), 500
        
        # Sanitize NaN values in the entire portfolio object
        sanitized_portfolio = sanitize_nan_values(portfolio)
        
        # Log success
        logger.info("Successfully generated portfolio")
        logger.debug(f"Number of assets in allocation: {len(sanitized_portfolio['asset_allocation'])}")
        logger.debug(f"Asset classes: {list(sanitized_portfolio['asset_allocation'].keys())}")
        
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
    required_fields = ['salary', 'investment_goals', 'risk_tolerance', 
                    'time_horizon', 'investment_amount']
    
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
    
    # Convert salary and investment_amount to float
    try:
        data['salary'] = float(data['salary'])
        data['investment_amount'] = float(data['investment_amount'])
    except ValueError:
        return "Salary and investment amount must be valid numbers"
    
    return None

@app.route('/portfolio/summary', methods=['GET'])
def get_portfolio_summary():
    """Get a summary of the generated portfolio."""
    if not portfolio_model:
        return jsonify({"error": "Portfolio model not initialized"}), 500
        
    summary = portfolio_model.get_summary()
    # Sanitize NaN values in the summary
    sanitized_summary = sanitize_nan_values(summary)
    return jsonify(sanitized_summary)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat messages from the user and return responses using the LLM.
    Integrates portfolio data to answer questions about allocations.
    """
    try:
        # Validate request
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message.strip():
            return jsonify({'response': 'Please enter a question about investing or your portfolio.'}), 400
        
        if not hf_client:
            logger.error("Hugging Face client not initialized")
            return jsonify({
                'response': "The Hugging Face API key is not configured. Please add your API key to the .env file."
            }), 400
        
        # Get portfolio data if it exists
        prompt = build_chat_prompt(user_message)
        
        # Generate response
        response = generate_chat_response(prompt)
        return jsonify({'response': response})
        
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({'response': "I'm having trouble answering right now. Please try again later."}), 500

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
Risk Profile: {portfolio_data.get('risk_profile', 'N/A')}
Risk Score: {portfolio_data.get('risk_score', 'N/A')}
Asset Allocation: 
- Stocks: {portfolio_data.get('asset_allocation', {}).get('stocks', 0) * 100:.1f}%
- Bonds: {portfolio_data.get('asset_allocation', {}).get('bonds', 0) * 100:.1f}%
- Crypto: {portfolio_data.get('asset_allocation', {}).get('crypto', 0) * 100:.1f}%
- Gold: {portfolio_data.get('asset_allocation', {}).get('gold', 0) * 100:.1f}%

Top Stock Picks: {', '.join([stock.get('ticker', '') for stock in portfolio_data.get('top_stocks', [])])}
Top Crypto Picks: {', '.join([crypto.get('ticker', '') for crypto in portfolio_data.get('top_crypto', [])])}

Predicted Returns:
- Overall: {portfolio_data.get('overall_predicted_return', 0) * 100:.2f}%
- Stocks: {portfolio_data.get('predicted_returns', {}).get('stocks', 0) * 100:.2f}%
- Bonds: {portfolio_data.get('predicted_returns', {}).get('bonds', 0) * 100:.2f}%
- Crypto: {portfolio_data.get('predicted_returns', {}).get('crypto', 0) * 100:.2f}%

Bond Strategy: {portfolio_data.get('bond_strategy', {}).get('method', 'N/A')} in a {portfolio_data.get('bond_strategy', {}).get('rate_environment', 'N/A')} rate environment

Gold Outlook:
- Market Score: {portfolio_data.get('gold_outlook', {}).get('market_score', 0):.2f}
- Market Fear: {portfolio_data.get('gold_outlook', {}).get('market_fear', 0):.2f}
```

Question: {user_message}

Answer (be concise and directly address the question based on the portfolio data):"""
    else:
        prompt = f"""You are a helpful investment advisor assistant. Provide clear, accurate, and helpful responses about investing, portfolio management, and financial planning.

Note: The user has not yet generated a portfolio recommendation, so you cannot provide specific details about their allocation.

Question: {user_message}

Answer (be concise and encourage the user to generate a portfolio if they're asking about specific allocations):"""
    
    return prompt

def generate_chat_response(prompt):
    """
    Generate a response using the Hugging Face client.
    
    Args:
        prompt: The prompt for the chat model
        
    Returns:
        str: The generated response
    """
    logger.info("Generating response using Hugging Face InferenceClient")
    try:
        response = hf_client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        # Clean up the response
        response_text = response.replace(prompt, "").strip()
        if not response_text:
            response_text = "I apologize, but I couldn't generate a proper response. Please try again."
            
        logger.info(f"Generated response: {response_text[:50]}...")
        return response_text
            
    except Exception as e:
        logger.error(f"Error during API request: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Error generating response: {str(e)}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(debug=True, host='127.0.0.1', port=port) 