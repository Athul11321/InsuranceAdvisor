from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
from datetime import datetime
import logging
import time
import re

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

class InsuranceChatbot:
    def __init__(self, model_path="insurance-chatbot-cpu"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
        # Add response cache for common questions
        self.response_cache = {}
        self.cache_hits = 0
        self.total_requests = 0
        
        # Quality thresholds
        self.min_response_length = 30
        self.max_response_length = 300
        
        # Company-specific information
        self.company_info = {
            'phone': '1-800-INSURANCE',
            'email': 'support@insurance.com',
            'website': 'www.insurance.com',
            'hours': '24/7',
            'app_name': 'Insurance Mobile App'
        }
        
    def load_model(self):   
        """Load the fine-tuned model with optimizations"""
        try:
            # Debug logging
            logger.info(f"Python path: {sys.path}")
            logger.info(f"Current directory: {os.getcwd()}")
            logger.info(f"Directory contents: {os.listdir('.')}")
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if we should use quantization
            use_quantization = os.environ.get('USE_QUANTIZATION', 'false').lower() == 'true'
            
            if use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    logger.info("Loading model with 8-bit quantization...")
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                except ImportError:
                    logger.warning("BitsAndBytes not available, loading without quantization")
                    use_quantization = False
            
            if not use_quantization:
                # Standard loading optimized for CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Preload common responses
            self.preload_common_responses()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M parameters")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def clean_response(self, response):
        """Remove template placeholders and improve response quality"""
        # Remove any remaining instruction formatting
        if "### Response:" in response:
            response = response.split("### Response:")[-1]
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0]
        
        # Remove common placeholders
        placeholders_to_remove = [
            r'\{\{[A-Z_]+\}\}',  # Matches {{ANYTHING}}
            r'\[\[.*?\]\]',      # Matches [[anything]]
            r'<[A-Z_]+>',        # Matches <ANYTHING>
            r'\{[A-Z_]+\}',      # Matches {ANYTHING}
        ]
        
        for pattern in placeholders_to_remove:
            response = re.sub(pattern, '', response)
        
        # Replace specific placeholders with actual values
        replacements = {
            '{{WEBSITE_URL}}': self.company_info['www.insurance.com'],
            '{{INSURANCE_SECTION}}': 'the insurance section',
            '{{PHONE_NUMBER}}': self.company_info['1-800-INSURANCE'],
            '{{EMAIL}}': self.company_info['support@insurance.com'],
            '{{COMPANY_NAME}}': 'our insurance company',
            '{{APP_NAME}}': self.company_info['insurance Mobile App'],
            'WEBSITE_URL': self.company_info['www.insurance.com'],
            'PHONE_NUMBER': self.company_info['1-800-INSURANCE'],
        }
        
        for placeholder, value in replacements.items():
            response = response.replace(placeholder, value)
        
        # Clean up formatting issues
        response = re.sub(r'\s+', ' ', response)  # Multiple spaces to single
        response = re.sub(r'\s*([.,!?])\s*', r'\1 ', response)  # Fix punctuation spacing
        response = re.sub(r'\.{2,}', '.', response)  # Multiple dots to single
        response = re.sub(r'\n{2,}', '\n', response)  # Multiple newlines to single
        
        # Fix numbered lists
        response = re.sub(r'(\d+)\s*\.\s*', r'\1. ', response)
        response = re.sub(r'(\d+)\s*\)\s*', r'\1) ', response)
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if sentences and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper sentence ending
        response = response.strip()
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response
    
    def validate_response_quality(self, response, user_input):
        """Check if response meets quality standards"""
        # Check length
        if len(response) < self.min_response_length:
            logger.warning(f"Response too short: {len(response)} chars")
            return False
        
        if len(response) > self.max_response_length:
            logger.warning(f"Response too long: {len(response)} chars")
            return False
        
        # Check for remaining placeholders
        if re.search(r'\{\{.*?\}\}|\[\[.*?\]\]|<[A-Z_]+>', response):
            logger.warning("Response contains placeholders")
            return False
        
        # Check for repetitive content
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # Too repetitive
                logger.warning(f"Response too repetitive: {unique_ratio:.2f} unique ratio")
                return False
        
        # Check for nonsensical patterns
        nonsense_patterns = [
            r'(.)\1{5,}',  # Same character repeated 5+ times
            r'(\w+\s+)\1{3,}',  # Same word repeated 3+ times
            r'[A-Z]{10,}',  # Too many consecutive capitals
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, response):
                logger.warning("Response contains nonsensical patterns")
                return False
        
        # Check relevance to insurance
        insurance_keywords = ['insurance', 'policy', 'claim', 'coverage', 'premium', 
                            'deductible', 'benefit', 'protection', 'accident', 'damage']
        user_keywords = user_input.lower().split()
        
        # If user asks about insurance, response should mention insurance concepts
        if any(keyword in user_input.lower() for keyword in insurance_keywords):
            if not any(keyword in response.lower() for keyword in insurance_keywords):
                logger.warning("Response not relevant to insurance query")
                return False
        
        return True
    
    def generate_response(self, user_input, max_length=150):
        """Generate response with quality checks and fallbacks"""
        if not self.model or not self.tokenizer:
            return self.get_fallback_response(user_input)
        
        self.total_requests += 1
        
        # Normalize input for cache lookup
        cache_key = user_input.lower().strip()
        cache_key = ' '.join(cache_key.split())  # Normalize whitespace
        
        # Check exact cache match
        if cache_key in self.response_cache:
            self.cache_hits += 1
            logger.info(f"Cache hit! ({self.cache_hits}/{self.total_requests})")
            
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_input,
                'bot': self.response_cache[cache_key],
                'cached': True
            })
            
            return self.response_cache[cache_key]
        
        # Check fuzzy cache match
        for cached_q, cached_response in self.response_cache.items():
            similarity = self.calculate_similarity(cache_key, cached_q)
            if similarity > 0.85:  # 85% similarity threshold
                self.cache_hits += 1
                logger.info(f"Fuzzy cache hit! Similarity: {similarity:.2f}")
                return cached_response
        
        # For critical insurance topics, prefer fallback responses
        critical_topics = {
            'claim': ['file', 'submit', 'process', 'claim', 'accident', 'damage'],
            'coverage': ['covered', 'coverage', 'protection', 'include', 'policy'],
            'payment': ['pay', 'payment', 'premium', 'cost', 'price', 'bill'],
            'emergency': ['emergency', 'urgent', 'immediate', 'accident', 'help']
        }
        
        input_lower = user_input.lower()
        for topic, keywords in critical_topics.items():
            if any(keyword in input_lower for keyword in keywords):
                fallback = self.get_enhanced_fallback_response(user_input, topic)
                if fallback:
                    logger.info(f"Using enhanced fallback for critical topic: {topic}")
                    return fallback
        
        # Try to generate with model
        prompt = f"""### Instruction:
        You are a helpful insurance assistant. Provide clear, accurate, and professional responses about insurance topics.
        User: {user_input}

    ### Response:"""
        
        try:
            # Generate with model
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("### Response:")[-1].strip()
            
            # Clean the response
            response = self.clean_response(response)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated response in {generation_time:.2f}s")
            
            # Validate response quality
            if not self.validate_response_quality(response, user_input):
                logger.warning("Generated response failed quality check, using fallback")
                response = self.get_fallback_response(user_input)
            
            # Cache good responses
            if len(cache_key) < 100 and self.validate_response_quality(response, user_input):
                self.response_cache[cache_key] = response
                logger.info(f"Cached response for: {cache_key[:30]}...")
            
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_input,
                'bot': response,
                'generation_time': generation_time,
                'cached': False,
                'source': 'model'
            })
            
            # Memory management
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-50:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self.get_fallback_response(user_input)
    
    def calculate_similarity(self, str1, str2):
        """Calculate similarity between two strings (0-1)"""
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_enhanced_fallback_response(self, user_input, topic):
        """Get enhanced fallback for critical topics"""
        enhanced_responses = {
            'claim': {
                'default': f"To file a claim, please follow these steps:\n\n1. **Immediate Action**: Call our 24/7 claims hotline at {self.company_info['phone']} or use our {self.company_info['app_name']}\n\n2. **Information Needed**:\n   - Your policy number\n   - Date, time, and location of incident\n   - Description of what happened\n   - Photos of any damage\n   - Police report number (if applicable)\n\n3. **Next Steps**: A claims adjuster will contact you within 24 hours to guide you through the process.\n\nNeed immediate assistance? Our claims team is available 24/7.",
                
                'status': "To check your claim status:\n\n1. **Online**: Log into your account at {self.company_info['website']}\n2. **Phone**: Call {self.company_info['phone']} with your claim number\n3. **Mobile App**: Track real-time updates on our {self.company_info['app_name']}\n\nMost claims are processed within 5-7 business days.",
                
                'time': "Claim processing times:\n\n• **Simple claims**: 3-5 business days\n• **Complex claims**: 7-14 business days\n• **Emergency repairs**: Authorized within 24 hours\n\nYou'll receive updates at each stage via email and SMS.",
            },
            'coverage': {
                'default': "Your insurance coverage includes:\n\n**Standard Coverage**:\n• Collision damage\n• Comprehensive (theft, vandalism, weather)\n• Liability protection\n• Medical payments\n\n**Optional Coverage**:\n• Rental car reimbursement\n• Roadside assistance\n• Gap insurance\n\nFor your specific coverage details, check your policy document or call {self.company_info['phone']}.",
                
                'what': "We offer several types of coverage:\n\n1. **Liability**: Covers damage you cause to others\n2. **Collision**: Covers your vehicle in accidents\n3. **Comprehensive**: Covers non-collision damage (theft, weather, vandalism)\n4. **Medical**: Covers medical expenses after an accident\n5. **Uninsured Motorist**: Protects you from uninsured drivers\n\nWhich type would you like to know more about?",
            },
            'payment': {
                'default': f"Payment options:\n\n• **Auto-Pay**: Set up automatic monthly payments\n• **Online**: Pay at {self.company_info['website']}\n• **Phone**: Call {self.company_info['phone']}\n• **Mobile App**: Use our {self.company_info['app_name']}\n• **Mail**: Send check to our payment processing center\n\nPayment is due by the date shown on your bill to avoid coverage lapse.",
                
                'premium': "Your premium is calculated based on:\n\n• **Driving record**: Claims and violations history\n• **Vehicle**: Make, model, year, and safety features\n• **Coverage levels**: Deductibles and limits you choose\n• **Location**: Where you live and park\n• **Discounts**: Multi-policy, safe driver, etc.\n\nFor a detailed breakdown of your premium, log into your account or call us.",
            },
            'emergency': {
                'default': f"**For emergencies**:\n\n🚨 **If you're in immediate danger, call 911 first**\n\n📞 Then call our 24/7 emergency line: {self.company_info['phone']}\n\n**We provide**:\n• Immediate claim initiation\n• Emergency towing coordination\n• Rental car arrangement\n• Direct billing to repair shops\n\nYour safety is our top priority. We're here to help 24/7.",
            }
        }
        
        # Find best matching response
        input_lower = user_input.lower()
        
        if topic in enhanced_responses:
            responses = enhanced_responses[topic]
            
            # Look for specific keywords to choose response
            for key, response in responses.items():
                if key != 'default' and key in input_lower:
                    return response.format(**self.company_info)
            
            # Return default for topic
            return responses['default'].format(**self.company_info)
        
        return None
    
    def preload_common_responses(self):
        """Preload cache with high-quality responses"""
        common_qa = {
            # Greetings
            "hello": "Hello! I'm your Insurance Assistant. I'm here to help you with claims, coverage questions, payments, and any other insurance needs. How can I assist you today?",
            "hi": "Hi there! I'm here to help with all your insurance questions. Whether you need to file a claim, check coverage, or understand your policy, I'm ready to assist. What can I help you with?",
            "help": "I can help you with:\n\n• **Claims**: File new claims or check status\n• **Coverage**: Understand what's covered\n• **Payments**: Payment options and billing\n• **Policy**: Make changes or get documents\n• **Quotes**: Get new coverage quotes\n\nWhat would you like assistance with?",
            
            # Claims
            "how do i file a claim": f"To file a claim:\n\n1. Call {self.company_info['phone']} (24/7) or use our mobile app\n2. Have your policy number ready\n3. Provide incident details and photos\n4. We'll assign an adjuster within 24 hours\n\nNeed to file a claim now? I can guide you through the process.",
            "claim status": f"To check your claim status:\n\n• **Online**: Log into {self.company_info['website']}\n• **Phone**: Call {self.company_info['phone']}\n• **App**: Real-time updates on our mobile app\n\nHave your claim number ready for faster service.",
            
            # Coverage
            "what does my insurance cover": "Your coverage typically includes:\n\n• **Liability**: Damage to others\n• **Collision**: Your vehicle damage\n• **Comprehensive**: Theft, weather, vandalism\n• **Medical**: Injury expenses\n\nFor your specific coverage, check your policy or I can help you understand any particular aspect.",
            "am i covered for": "To check if you're covered, I'll need to know what specific situation you're asking about. Common coverages include:\n\n• Accidents and collisions\n• Theft and vandalism\n• Weather damage\n• Medical expenses\n• Rental car reimbursement\n\nWhat specific coverage are you wondering about?",
            
            # Deductibles
            "what is a deductible": "A deductible is your out-of-pocket cost before insurance pays. For example:\n\nIf you have a $500 deductible and $2,000 in damage:\n• You pay: $500\n• Insurance pays: $1,500\n\nHigher deductibles = lower premiums\nLower deductibles = higher premiums\n\nWould you like to know your current deductibles?",
            "what is my deductible": f"To find your deductible amounts:\n\n1. Check your policy documents\n2. Log into {self.company_info['website']}\n3. Call {self.company_info['phone']}\n4. Check our mobile app\n\nYou may have different deductibles for collision and comprehensive coverage.",
            
            # Payments/Premiums
            "how much is my premium": f"To check your premium:\n\n• **Online**: Log into {self.company_info['website']}\n• **Phone**: Call {self.company_info['phone']}\n• **App**: View in our mobile app\n• **Documents**: Check your policy documents\n\nWant to know about ways to lower your premium?",
            "payment options": f"We offer flexible payment options:\n\n• **Auto-Pay**: Automatic monthly deductions\n• **Online**: Pay at {self.company_info['website']}\n• **App**: Quick payments via mobile\n• **Phone**: Call {self.company_info['phone']}\n• **Mail**: Check payments accepted\n\nNeed to update your payment method?",
            
            # Policy changes
            "cancel policy": f"To cancel your policy:\n\n1. Call {self.company_info['phone']} to speak with an agent\n2. Provide your policy number and reason\n3. Confirm your cancellation date\n4. Receive confirmation in writing\n\n⚠️ Make sure you have new coverage before canceling to avoid gaps.",
            "change coverage": f"To modify your coverage:\n\n• **Online**: Update at {self.company_info['website']}\n• **Phone**: Call {self.company_info['phone']}\n• **Agent**: Contact your local agent\n\nChanges typically take effect immediately or at your next billing cycle.",
        }
        
        # Convert all keys to lowercase and add variations
        expanded_cache = {}
        for key, value in common_qa.items():
            expanded_cache[key.lower()] = value
            # Add variations
            if "how do i" in key:
                expanded_cache[key.replace("how do i", "how can i")] = value
                expanded_cache[key.replace("how do i", "how to")] = value
            if "what is" in key:
                expanded_cache[key.replace("what is", "what's")] = value
                expanded_cache[key.replace("what is", "whats")] = value
        
        self.response_cache.update(expanded_cache)
        logger.info(f"Preloaded {len(expanded_cache)} common responses")
    
    def get_fallback_response(self, user_input):
        """Provide fallback responses for common queries"""
        fallback_responses = {
            'claim': f"To file a claim: Call {self.company_info['phone']} (24/7) or use our mobile app. Have your policy number and incident details ready. We'll guide you through the process and assign an adjuster quickly.",
            'premium': "Your premium depends on coverage type, vehicle, driving history, and location. Check your policy document or account for specific amounts. We offer various discounts that might lower your premium.",
            'coverage': "Insurance coverage varies by policy. Common types include liability, collision, comprehensive, and medical. Check your policy document or contact us for your specific coverage details.",
            'deductible': "A deductible is what you pay before insurance covers the rest. For example, with a $500 deductible on $2,000 damage, you pay $500. Check your policy for your specific deductibles.",
            'payment': f"Pay your premium online at {self.company_info['website']}, through our app, by phone at {self.company_info['phone']}, or by mail. Set up auto-pay to never miss a payment.",
            'contact': f"Contact us 24/7 at {self.company_info['phone']}, visit {self.company_info['website']}, or use our mobile app. For emergencies, we're always available to help.",
            'policy': f"Access your policy documents online at {self.company_info['website']} or through our mobile app. For changes or questions, call {self.company_info['phone']}.",
        }
        
        user_input_lower = user_input.lower()
        
        # Check each keyword
        for key, response in fallback_responses.items():
            if key in user_input_lower:
                return response
        
        # Default response
        return f"I'm here to help with your insurance needs. You can ask about:\n\n• Filing claims\n• Coverage details\n• Premium payments\n• Policy information\n• Deductibles\n\nFor immediate assistance, call {self.company_info['phone']} (24/7). How can I help you today?"

# Initialize chatbot
chatbot = InsuranceChatbot()

@app.route('/')
def index():
    return render_template('index.html')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Log the request
        logger.info(f"User message: {user_message}")
        
        # Generate response
        response = chatbot.generate_response(user_message)
        
        # Log the response source
        if chatbot.conversation_history:
            last_entry = chatbot.conversation_history[-1]
            logger.info(f"Response source: {last_entry.get('source', 'unknown')}, Cached: {last_entry.get('cached', False)}")
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'history': chatbot.conversation_history})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.model is not None,
        'cache_size': len(chatbot.response_cache),
        'company_info': chatbot.company_info
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Get chatbot performance metrics"""
    total_conversations = len(chatbot.conversation_history)
    cache_ratio = (chatbot.cache_hits / chatbot.total_requests * 100) if chatbot.total_requests > 0 else 0
    
    # Calculate average response time
    response_times = [
        conv.get('generation_time', 0) 
        for conv in chatbot.conversation_history 
        if not conv.get('cached', False) and conv.get('generation_time', 0) > 0
    ]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Source breakdown
    sources = {'model': 0, 'fallback': 0, 'cache': 0}
    for conv in chatbot.conversation_history:
        if conv.get('cached'):
            sources['cache'] += 1
        elif conv.get('source') == 'model':
            sources['model'] += 1
        else:
            sources['fallback'] += 1
    
    # Memory usage estimation
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
    except:
        memory_usage_mb = 0
    
    return jsonify({
        'total_conversations': total_conversations,
        'cache_hits': chatbot.cache_hits,
        'total_requests': chatbot.total_requests,
        'cache_hit_ratio': f"{cache_ratio:.1f}%",
        'average_response_time': f"{avg_response_time:.2f}s",
        'model_loaded': chatbot.model is not None,
        'cache_size': len(chatbot.response_cache),
        'memory_usage_mb': f"{memory_usage_mb:.1f}",
        'response_sources': sources
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system status"""
    return jsonify({
        'model_path': chatbot.model_path,
        'model_loaded': chatbot.model is not None,
        'tokenizer_loaded': chatbot.tokenizer is not None,
        'cache_entries': list(chatbot.response_cache.keys())[:10],  # First 10 cache keys
        'recent_conversations': len(chatbot.conversation_history),
        'company_info': chatbot.company_info
    })

if __name__ == '__main__':
    # Model path detection
    possible_paths = [
        "./insurance-chatbot-cpu",
        "./insurance-chatbot-model-cpu",
        "./insurance-chatbot-model-cpu/checkpoint-1900",
        "./insurance-chatbot-model-cpu/checkpoint-1950",
        "./insurance-chatbot-model-cpu/checkpoint-1800",
        "./models/insurance-chatbot-final",
        "./models/insurance-chatbot-cpu",
        "./DialoGPT-medium-insurance"  # Add your actual model path
    ]
    
    model_found = False
    for path in possible_paths:
        if os.path.exists(path):
            if os.path.exists(os.path.join(path, "config.json")):
                chatbot.model_path = path
                logger.info(f"Found model at: {path}")
                if chatbot.load_model():
                    model_found = True
                    break
    
    if not model_found:
        logger.warning("Model not found in any expected location. Running in fallback mode.")
        logger.info("The chatbot will use high-quality predefined responses.")
        logger.info("\nExpected model locations:")
        for path in possible_paths:
            logger.info(f"  - {path}")
        logger.info("\nCurrent directory contents:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                logger.info(f"  Directory: {item}")
                try:
                    sub_items = os.listdir(item)
                    if "config.json" in sub_items:
                        logger.info(f"    -> Contains model files!")
                except:
                    pass
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)