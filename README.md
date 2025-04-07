# PortfolioOptimizer

A web application that generates personalized investment portfolios based on user preferences and market data.

## Overview

This application:
- Takes user input including risk tolerance, investment amount, and time horizon
- Generates personalized portfolio allocations across stocks, bonds, cryptocurrencies, and gold
- Provides detailed breakdowns of specific investments with expected returns
- Offers a chat interface for discussing the portfolio and investment strategies

## Project Structure

```
src/
├── config/           # Configuration settings
├── models/           # Portfolio generation models
├── utils/            # Utility modules
│   ├── cache_manager.py      # Caching with entry-level expiration (1 week max)
│   └── data_service.py       # Financial data service with retry logic
├── web/              # Web application
│   ├── app.py        # Flask application with API endpoints
│   ├── static/       # Static web assets (JS, CSS, images)
│   └── templates/    # HTML templates
└── test_data_service.py  # Tests for data service
```

## Technical Features

- **Real-time Financial Data**: Uses yfinance API to fetch current market data
- **Efficient Caching**: Implements caching with per-entry expiration (max 1 week)
- **Rate Limit Handling**: Includes backoff retry mechanism for API rate limits
- **NaN Value Handling**: Properly sanitizes NaN values in JSON responses
- **LLM Integration**: Uses Hugging Face Inference API for conversational responses

## Dependencies

- Python 3.8+
- Flask - Web framework
- yfinance - Financial data API
- pandas - Data manipulation
- numpy - Numerical processing
- HuggingFace Inference API - LLM for chat

## Environment Variables

Required environment variables (set in .env file):
```
SECRET_KEY=your_flask_secret_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
API_TIMEOUT=30
MAX_PORTFOLIO_SIZE=25
PORT=8080
```

## Running the Application

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with the required environment variables
4. Run the application:
   ```
   python src/web/app.py
   ```
5. Navigate to http://localhost:8080 in your browser

## Testing

Run the data service tests:
```
python src/test_data_service.py
```

## API Endpoints

- `GET /` - Home page
- `POST /portfolio` - Generate a portfolio based on user inputs
- `GET /portfolio/summary` - Get a summary of the generated portfolio
- `POST /chat` - Chat interface for portfolio questions

## Development Notes

- The financial data service implements exponential backoff for API rate limiting
- All cache entries expire after a maximum of one week
- Error handling returns empty structures rather than failing
- Input validation is performed for all API requests 