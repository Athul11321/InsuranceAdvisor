// Global variables
let isTyping = false;

// Send message function
async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (!message || isTyping) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input
    input.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    isTyping = true;
    
    try {
        // Send message to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        hideTypingIndicator();
        isTyping = false;
        
        if (response.ok) {
            // Add bot response
            addMessage(data.response, 'bot');
        } else {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        }
    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        isTyping = false;
        addMessage('Sorry, I couldn\'t connect to the server. Please try again.', 'bot');
    }
}

// Add message to chat
function addMessage(message, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'bot' ? '🤖' : '👤';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${escapeHtml(message)}</p>`;
    
    if (sender === 'bot') {
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
    } else {
        messageDiv.appendChild(content);
        messageDiv.appendChild(avatar);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'message bot-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Quick action function
function quickAction(message) {
    document.getElementById('user-input').value = message;
    sendMessage();
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Format bot responses with better styling
function formatBotResponse(response) {
    // Convert numbered lists
    response = response.replace(/(\d+)\.\s/g, '<br>$1. ');
    
    // Convert bullet points
    response = response.replace(/•\s/g, '<br>• ');
    
    // Add line breaks for readability
    response = response.replace(/\n/g, '<br>');
    
    return response;
}

// Load chat history on page load
window.addEventListener('load', async function() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            addMessage('Note: The AI model is not loaded. I\'ll use fallback responses to help you.', 'bot');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});

// Add enter key support for input
document.getElementById('user-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

// Add feedback buttons to each message
function addFeedbackButtons(messageDiv, messageIndex) {
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'feedback-buttons';
    feedbackDiv.innerHTML = `
        <button onclick="sendFeedback('positive', ${messageIndex})" class="feedback-btn positive">
            👍 Helpful
        </button>
        <button onclick="sendFeedback('negative', ${messageIndex})" class="feedback-btn negative">
            👎 Not Helpful
        </button>
    `;
    messageDiv.appendChild(feedbackDiv);
}

async function sendFeedback(type, index) {
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                type: type,
                index: index 
            })
        });
        
        if (response.ok) {
            // Visual feedback
            document.querySelectorAll('.feedback-buttons')[index]
                .innerHTML = '<span class="feedback-sent">✅ Thank you for your feedback!</span>';
        }
    } catch (error) {
        console.error('Error sending feedback:', error);
    }
}