import re
import random
import json
import os
from datetime import datetime

class PortfolioAssistant:
    """
    A chatbot implementation to explain portfolio recommendations and
    answer user questions about investments.
    """
    
    def __init__(self):
        """Initialize the portfolio assistant."""
        self.portfolio_data = None
        self.user_profile = None
        self.conversation_history = []
        
        # Load predefined responses
        self.responses = self._load_responses()
    
    def _load_responses(self):
        """
        Load predefined responses from a JSON file if available,
        otherwise use default responses.
        
        Returns:
            dict: Dictionary of response templates.
        """
        try:
            # Try to load responses from file
            responses_path = os.path.join(os.path.dirname(__file__), 'responses.json')
            if os.path.exists(responses_path):
                with open(responses_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading responses: {e}")
        
        # Default responses
        return {
            "greeting": [
                "Hello! I'm your portfolio assistant. How can I help you with your investment portfolio today?",
                "Welcome! I'm here to help with your investment questions and portfolio recommendations.",
                "Hi there! I can help explain your portfolio allocation or answer investment questions."
            ],
            "farewell": [
                "Goodbye! Feel free to return if you have more investment questions.",
                "Thanks for chatting. I'm here whenever you need investment advice!",
                "Have a great day! Don't hesitate to ask if you have more questions about your portfolio."
            ],
            "fallback": [
                "I'm not sure I understand. Could you rephrase your question about your portfolio or investments?",
                "I don't have enough information to answer that. Could you ask something specific about your portfolio allocation or investment strategy?",
                "I'm still learning. Could you ask me something about your risk profile, asset allocation, or specific investments in your portfolio?"
            ]
        }
    
    def set_portfolio_data(self, portfolio_data):
        """
        Set the portfolio data for the assistant to reference.
        
        Args:
            portfolio_data (dict): Portfolio recommendation data.
        """
        self.portfolio_data = portfolio_data
        if portfolio_data and 'user_profile' in portfolio_data:
            self.user_profile = portfolio_data['user_profile']
    
    def add_message_to_history(self, role, message):
        """
        Add a message to the conversation history.
        
        Args:
            role (str): 'user' or 'bot'
            message (str): The message content
        """
        self.conversation_history.append({
            'role': role,
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def process_message(self, message):
        """
        Process a user message and generate a response.
        
        Args:
            message (str): User message.
            
        Returns:
            str: Assistant response.
        """
        # Add user message to history
        self.add_message_to_history('user', message)
        
        # Generate simple response for now
        response = "Thank you for your message. The portfolio assistant functionality is under development."
        
        # Add response to history
        self.add_message_to_history('bot', response)
        
        return response 