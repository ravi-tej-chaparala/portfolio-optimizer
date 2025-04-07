document.addEventListener('DOMContentLoaded', function() {
    // Chatbot UI elements
    const chatbotContainer = document.querySelector('.chatbot-container');
    const chatbotHeader = document.getElementById('chatbotHeader');
    const chatMessages = document.getElementById('chatMessages');
    const userMessageInput = document.getElementById('userMessage');
    const sendMessageBtn = document.getElementById('sendMessageBtn');
    const toggleChatBtn = document.getElementById('toggleChatBtn');
    const closeChatBtn = document.getElementById('closeChatBtn');
    
    // Initialize chat state
    chatbotContainer.classList.remove('active');
    toggleChatBtn.style.display = 'flex';
    
    // Function to toggle chatbot visibility
    function toggleChatbot() {
        chatbotContainer.classList.toggle('active');
        if (chatbotContainer.classList.contains('active')) {
            toggleChatBtn.style.display = 'none';
            userMessageInput.focus();
        } else {
            toggleChatBtn.style.display = 'flex';
        }
    }
    
    // Toggle chat window
    toggleChatBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        chatbotContainer.classList.add('active');
        toggleChatBtn.style.display = 'none';
        userMessageInput.focus();
    });
    
    // Close chat window
    closeChatBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        chatbotContainer.classList.remove('active');
        toggleChatBtn.style.display = 'flex';
    });
    
    // Send message on Enter (but allow Shift+Enter for new lines)
    userMessageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send message on button click
    sendMessageBtn.addEventListener('click', sendMessage);
    
    // Function to add a message to the chat
    function addMessage(text, role, isHtml = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isHtml) {
            contentDiv.innerHTML = text;
        } else {
            contentDiv.textContent = text;
        }
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to add a suggestion button
    function addSuggestionButton(text) {
        const button = document.createElement('button');
        button.className = 'suggestion-btn';
        button.textContent = text;
        button.addEventListener('click', function() {
            userMessageInput.value = text;
            sendMessage();
        });
        return button;
    }
    
    // Function to add suggestion buttons
    function addSuggestions(suggestions) {
        const suggestionsDiv = document.createElement('div');
        suggestionsDiv.className = 'message assistant suggestions';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<p>You can ask me:</p>';
        
        const buttonsDiv = document.createElement('div');
        buttonsDiv.className = 'suggestion-buttons';
        
        suggestions.forEach(suggestion => {
            buttonsDiv.appendChild(addSuggestionButton(suggestion));
        });
        
        contentDiv.appendChild(buttonsDiv);
        suggestionsDiv.appendChild(contentDiv);
        chatMessages.appendChild(suggestionsDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to send a message
    function sendMessage() {
        const message = userMessageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        userMessageInput.value = '';
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant';
        loadingDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Send to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            loadingDiv.remove();
            // Add assistant response
            addMessage(data.response, 'assistant');
        })
        .catch(error => {
            console.error('Error:', error);
            // Remove loading indicator
            loadingDiv.remove();
            // Add error message
            addMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        });
    }
    
    // Listen for portfolio generation completion event to add portfolio-specific suggestions
    document.addEventListener('portfolioGenerated', function() {
        // Only add suggestions if chat is currently not in use
        if (chatMessages.children.length <= 1) {
            const portfolioSuggestions = [
                "Why did I get this asset allocation?",
                "Explain my risk profile",
                "What are my top stock picks?",
                "Tell me about my bond strategy",
                "What's my predicted portfolio return?",
                "How much is allocated to crypto?",
                "Should I invest more in gold?"
            ];
            
            // Clear previous messages and add welcome message
            chatMessages.innerHTML = '';
            addMessage("Your portfolio has been generated! How can I help you understand your investment recommendations?", "assistant");
            
            // Add portfolio-specific suggestions
            addSuggestions(portfolioSuggestions);
        } else {
            // If the chat is already in use, just add a notification
            addMessage("I notice your portfolio has been generated! You can now ask me specific questions about your investment recommendations.", "assistant");
        }
    });
    
    // Initial suggestions for users before portfolio generation
    const initialSuggestions = [
        "What factors affect asset allocation?",
        "How do I determine my risk tolerance?",
        "What's the difference between stocks and bonds?",
        "Should I invest in cryptocurrency?",
        "What is dollar-cost averaging?"
    ];
    
    // Add initial suggestions
    addSuggestions(initialSuggestions);
}); 