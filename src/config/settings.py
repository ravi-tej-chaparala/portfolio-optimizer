"""
Global settings and configuration for the investment portfolio application.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Asset symbols used throughout the application
ASSETS = {
    'stocks': {
        'index': '^GSPC',  # S&P 500
        'sectors': {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'V'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Communication Services': ['NFLX', 'DIS', 'CMCSA', 'VZ', 'T'],
            'Industrial': ['HON', 'UNP', 'UPS', 'CAT', 'BA'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
            'Basic Materials': ['LIN', 'APD', 'ECL', 'NEM', 'FCX'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA']
        },
        'fallback': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JNJ', 'PG', 'V']
    },
    'crypto': {
        'main': ['BTC-USD', 'ETH-USD'],
        'altcoins': ['SOL-USD', 'ADA-USD', 'XRP-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD'],
        'fallback': ['BTC-USD', 'ETH-USD', 'SOL-USD']
    },
    'bonds': {
        'short_term': ['SHY', 'VGSH', 'BSV'],
        'intermediate': ['IEF', 'VGIT', 'BIV'],
        'long_term': ['TLT', 'VGLT', 'BLV'],
        'treasury_rates': {
            'short': '^IRX',  # 13-week Treasury bill
            'long': '^TNX'    # 10-Year Treasury
        },
        'fallback': ['AGG', 'BND', 'VCIT']
    },
    'gold': {
        'etfs': ['GLD', 'IAU', 'SGOL', 'BAR', 'GLDM'],
        'leveraged': ['UGL', 'GDX'],
        'indicator': 'GC=F',  # Gold Futures
        'fallback': ['GLD', 'SGOL']
    }
}

# API settings
API_SETTINGS = {
    'yfinance': {
        'max_requests_per_hour': 60,
        'request_delay': 1.0,
    }
}

# Cache settings
CACHE_SETTINGS = {
    'ttl': 2592000,  # Time to live in seconds (30 days)
    'force_refresh': False
}

# Risk profile mappings
RISK_PROFILES = {
    'conservative': {
        'allocation': {
            'stocks': 0.30,
            'bonds': 0.50,
            'crypto': 0.05,
            'gold': 0.15
        },
        'score': 0.25
    },
    'moderately_conservative': {
        'allocation': {
            'stocks': 0.40,
            'bonds': 0.40,
            'crypto': 0.10,
            'gold': 0.10
        },
        'score': 0.40
    },
    'moderate': {
        'allocation': {
            'stocks': 0.50,
            'bonds': 0.30,
            'crypto': 0.15,
            'gold': 0.05
        },
        'score': 0.55
    },
    'aggressive': {
        'allocation': {
            'stocks': 0.60,
            'bonds': 0.20,
            'crypto': 0.15,
            'gold': 0.05
        },
        'score': 0.75
    },
    'very_aggressive': {
        'allocation': {
            'stocks': 0.70,
            'bonds': 0.10,
            'crypto': 0.15,
            'gold': 0.05
        },
        'score': 0.90
    }
} 