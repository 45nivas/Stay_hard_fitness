"""
Fitness Chatbot using Gemini API for personalized fitness guidance
Handles user queries about workouts, nutrition, and fitness goals
"""

import re
import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class FitnessChatbot:
    def __init__(self):
        self.user_data = {}
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        # Keep Gemini as fallback
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    def calculate_daily_nutrition(self, user_profile_data):
        """Calculate proper daily nutrition for gym goals - FIXED VERSION"""
        if not user_profile_data:
            return "Please set up your profile first for personalized nutrition advice!"
        
        try:
            weight_kg = float(user_profile_data.get('weight', 70))
            height_cm = float(user_profile_data.get('height', 170))
            age = int(user_profile_data.get('age', 25))
            gender = user_profile_data.get('gender', 'Male')
            goal = user_profile_data.get('primary_goal_code', 'muscle_gain')
            
            # Calculate BMR using Mifflin-St Jeor equation
            if gender.lower() == 'male':
                bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
            else:
                bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
            
            # Activity multiplier for gym users
            tdee = bmr * 1.6
            
            # Adjust calories based on fitness goal
            if goal == 'muscle_gain':
                daily_calories = tdee + 400  # Surplus for bulking
                protein_per_kg = 2.2  # Higher protein for muscle gain
            elif goal == 'weight_loss':
                daily_calories = tdee - 500  # Deficit for cutting
                protein_per_kg = 2.4  # Higher protein to preserve muscle
            else:
                daily_calories = tdee  # Maintenance
                protein_per_kg = 2.0
            
            # Calculate macros CORRECTLY (this was the bug!)
            protein_grams = weight_kg * protein_per_kg
            protein_calories = protein_grams * 4  # 4 calories per gram
            
            fat_calories = daily_calories * 0.25  # 25% of total calories
            fat_grams = fat_calories / 9  # 9 calories per gram of fat (NOT calories = grams!)
            
            carb_calories = daily_calories - protein_calories - fat_calories
            carb_grams = carb_calories / 4  # 4 calories per gram of carbs (NOT calories = grams!)
            
            # Format response for gym users
            response = f"""🏋️ **Your Personalized Nutrition Plan:**

**Daily Target:** {daily_calories:.0f} calories
**BMR:** {bmr:.0f} calories | **TDEE:** {tdee:.0f} calories

**Macronutrient Breakdown:**
💪 **Protein:** {protein_grams:.0f}g ({protein_calories:.0f} calories)
🍞 **Carbs:** {carb_grams:.0f}g ({carb_calories:.0f} calories)  
🥑 **Fat:** {fat_grams:.0f}g ({fat_calories:.0f} calories)

**Fitness Goal:** {goal.replace('_', ' ').title()}
**Protein Target:** {protein_per_kg}g per kg body weight

💡 **Pro Tips:**
- Use our voice calorie tracker to log meals
- Eat protein within 30 mins post-workout
- Stay hydrated: {weight_kg * 35:.0f}ml water daily
- Track progress with our pose detection workouts!"""

            return response
            
        except (ValueError, KeyError) as e:
            return f"⚠️ Unable to calculate nutrition. Please check your profile data. Error: {str(e)}"

    def provide_fat_loss_advice(self, message, user_profile_data):
        """Provide specific fat loss advice, extracting stats from message when possible"""
        import re
        
        # Try to extract stats from the message
        weight_match = re.search(r'(\d+)\s*kg', message.lower())
        height_match = re.search(r'(\d+)\s*(?:cm|height)', message.lower())
        calorie_match = re.search(r'(\d+)\s*(?:calories|kcal)', message.lower())
        
        # Use extracted data or defaults
        weight = int(weight_match.group(1)) if weight_match else 70
        height = int(height_match.group(1)) if height_match else 170
        current_calories = int(calorie_match.group(1)) if calorie_match else 2000
        
        # Calculate BMR and TDEE for fat loss (assuming male, age 25)
        bmr = (10 * weight) + (6.25 * height) - (5 * 25) + 5
        tdee = bmr * 1.6  # Active lifestyle
        fat_loss_calories = tdee - 500  # 500 calorie deficit
        
        return f"""🔥 **Fat Loss Plan for {weight}kg, {height}cm:**

**Current Status:**
- Your TDEE: ~{tdee:.0f} calories
- You're eating: {current_calories} calories
- For fat loss: {fat_loss_calories:.0f} calories (-500 deficit)

**📊 Fat Loss Recommendations:**
💡 **Calorie Target:** {fat_loss_calories:.0f} calories/day
🥩 **Protein:** {weight * 2.4:.0f}g (preserve muscle)
🍞 **Carbs:** {fat_loss_calories * 0.25 / 4:.0f}g (energy for workouts)
🥑 **Fat:** {fat_loss_calories * 0.25 / 9:.0f}g (hormone production)

**🏃‍♂️ Action Steps:**
1. **Reduce** to {fat_loss_calories:.0f} calories from {current_calories}
2. **High protein** to maintain muscle mass
3. **Strength training** 3-4x/week (use our app!)
4. **Track everything** with our voice calorie tracker

**Pro Tips:**
- 1kg fat = 7700 calories deficit
- Expect 0.5-1kg loss per week
- Use our app's pose detection for perfect form! 💪"""

    def process_message(self, message, user_profile_data=None):
        """Process user message with intelligent routing and fitness focus"""
        message_lower = message.lower()
        
        # Sleep and lifestyle queries - prioritize these
        if any(keyword in message_lower for keyword in ['sleep', 'rest', 'recovery', 'tired', 'time management', 'busy', 'schedule']):
            return self.generate_general_response(message, user_profile_data)
        
        # Nutrition queries - check for fat loss specific questions first
        elif any(keyword in message_lower for keyword in ['lose fat', 'fat loss', 'cutting', 'lose weight']):
            return self.provide_fat_loss_advice(message, user_profile_data)
        # More specific nutrition keywords to avoid false triggers
        elif any(keyword in message_lower for keyword in ['daily calories', 'how many calories', 'nutrition plan', 'macros', 'diet plan', 'bmr', 'tdee', 'meal plan', 'what should i eat', 'protein intake', 'carb intake']):
            return self.calculate_daily_nutrition(user_profile_data)
        
        # Workout queries with pose detection context
        elif any(keyword in message_lower for keyword in ['workout', 'exercise', 'squat', 'pushup', 'curl', 'rep']):
            return self.generate_workout_response(message, user_profile_data)
        
        # Non-fitness questions but with redirection - use word boundaries for precise matching
        elif any(re.search(r'\b' + keyword + r'\b', message_lower) for keyword in ['language model', 'llm', 'ai', 'technology', 'computer', 'programming']):
            return self.handle_non_fitness_with_redirect(message)
        
        # General fitness chat
        else:
            return self.generate_general_response(message, user_profile_data)
    
    def handle_non_fitness_with_redirect(self, message):
        """Handle non-fitness questions intelligently then redirect to fitness"""
        message_lower = message.lower()
        
        # Large Language Model / AI questions - use word boundaries
        if any(re.search(r'\b' + keyword + r'\b', message_lower) for keyword in ['language model', 'llm', 'ai']):
            return """🤖 **That's a great question about AI!**

A Large Language Model (LLM) is an AI system trained on vast amounts of text data to understand and generate human-like responses. They're used in applications like chatbots, content creation, and code assistance.

💪 **Speaking of AI - that's exactly what powers this fitness app!**

Our chatbot uses advanced AI to:
- 🎯 Give personalized workout advice
- 📊 Calculate your nutrition needs  
- 🎤 Process voice commands for calorie tracking
- 🏋️ Provide real-time exercise guidance

**Try asking me:**
- "What's my daily calorie goal?"
- "Give me a workout routine"
- "How do I track my meals?"

Let's focus on your fitness journey! What are your current fitness goals? 💪"""

        # Technology/Programming questions
        elif any(keyword in message_lower for keyword in ['technology', 'computer', 'programming', 'code']):
            return """💻 **Interesting tech question!**

Technology is fascinating, especially when it's applied to solve real-world problems like fitness and health tracking!

🚀 **This app showcases modern tech:**
- **Django Backend:** Robust web framework
- **MediaPipe AI:** Real-time pose detection
- **Voice Recognition:** AI-powered calorie tracking
- **Computer Vision:** Exercise form analysis
- **Machine Learning:** Personalized recommendations

💪 **Let's put this tech to work for your fitness!**

**Available Features:**
- Voice calorie tracking (just speak your meals!)
- Real-time workout form correction
- Personalized nutrition calculations

What fitness goal would you like to work on today? 🏋️"""

        # General non-fitness topics
        else:
            return """🤔 **That's an interesting topic!**

While I'd love to discuss that, I'm specifically designed to be your **AI fitness coach** and help you achieve your health goals.

💪 **I'm here to help you with:**
- 🏋️ Personalized workout routines
- 🍎 Nutrition and calorie tracking  
- 📊 Fitness goal planning
- 🎯 Exercise form and technique
- 📱 Using our app's advanced features

**Popular questions I can help with:**
- "What should I eat to gain muscle?"
- "How many calories do I need daily?"
- "What's a good beginner workout?"
- "How do I use the voice calorie tracker?"

Let's focus on your fitness journey! What would you like to know about health and exercise? 💪"""
    
    def generate_workout_response(self, message, user_profile_data):
        """Generate workout responses with MediaPipe pose detection context"""
        
        prompt = f"""You are a fitness trainer for a gym app with OpenCV pose detection.
        
User message: "{message}"
User profile: {user_profile_data}

Available workouts with real-time pose correction:
- Squats (MediaPipe landmark tracking)
- Push-ups (Form analysis with OpenCV)
- Bicep Curls (Rep counting)
- Hammer Curls (Left/right arm tracking)
- Side Raises (Shoulder form check)

Provide specific, actionable fitness advice. Mention our pose detection features when relevant.
Keep responses under 200 words and gym-focused."""

        return self._query_ollama(prompt)
    
    def generate_general_response(self, message, user_profile_data):
        """Generate general fitness responses"""
        
        prompt = f"""You are an AI fitness trainer for a modern gym app.
        
User message: "{message}"
User profile: {user_profile_data}

Features available:
- Real-time pose correction with MediaPipe
- Voice calorie tracking
- Rep counting for 5 exercises
- One rep max calculator

Be encouraging, knowledgeable, and reference our app features when helpful.
Keep responses under 150 words."""

        return self._query_ollama(prompt)
    
    def _query_ollama(self, prompt):
        """Query Ollama local LLM for fitness responses with fallback"""
        try:
            payload = {
                "model": "llama3.2",  # Default model, can be changed
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            print(f"DEBUG: Calling Ollama with prompt: {prompt[:100]}...")
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=15  # Ollama might be slower locally
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('response', '').strip()
                print(f"DEBUG: Ollama responded with: {result[:100]}...")
                return result
            else:
                print(f"Ollama error: {response.status_code}")
                return self._get_offline_response(prompt)
                
        except requests.exceptions.ConnectionError:
            print("DEBUG: Ollama not running - using offline responses")
            return self._get_offline_response(prompt)
        except Exception as e:
            print(f"DEBUG: Ollama error: {e}")
            return self._get_offline_response(prompt)
    
    def _get_offline_response(self, prompt):
        """Provide built-in responses for demo when API is unavailable"""
        prompt_lower = prompt.lower()
        
        # Check if this is a non-fitness question being processed - use word boundaries
        if any(re.search(r'\b' + word + r'\b', prompt_lower) for word in ['language model', 'llm', 'ai', 'technology', 'computer']):
            # For non-fitness questions, use the redirect method
            if re.search(r'\b(language model|llm)\b', prompt_lower):
                return self.handle_non_fitness_with_redirect("what is large language model")
            elif re.search(r'\b(technology|computer)\b', prompt_lower):
                return self.handle_non_fitness_with_redirect("technology question")
            else:
                # Handle general AI questions or fallback
                return self.handle_non_fitness_with_redirect("ai question")
        
        # Workout-related responses
        elif any(word in prompt_lower for word in ['workout', 'exercise', 'squat', 'pushup', 'curl']):
            return """💪 **Great question about workouts!**

Here are some effective exercises you can try:

🏋️ **Upper Body:**
- Push-ups (our app tracks your form!)
- Bicep curls with pose detection
- Hammer curls for forearm strength

🦵 **Lower Body:**
- Squats with real-time form correction
- Side raises for shoulder stability

📱 **App Features:**
- Real-time pose correction with MediaPipe
- Rep counting for all exercises
- Form analysis to prevent injury

Start with 3 sets of 8-12 reps. Our pose detection will help you maintain perfect form!"""

        # Nutrition-related responses  
        elif any(word in prompt_lower for word in ['nutrition', 'calories', 'diet', 'protein']):
            return """🍎 **Nutrition is key to your fitness success!**

**General Guidelines:**
- Eat 1.6-2.2g protein per kg body weight
- Stay hydrated: 35ml water per kg body weight
- Eat within 30 mins post-workout

📱 **Use Our Voice Tracker:**
- Say "I ate 25g olive oil" (225 calories)
- Real-time nutrition tracking
- Accurate calorie calculations

**Pro Tip:** Use our calorie tracker to log your meals easily. Just speak into the microphone and let AI do the rest!"""

        # General fitness responses
        else:
            return """💪 **Welcome to your fitness journey!**

**Our App Features:**
🎯 Real-time pose detection for perfect form
🎤 Voice-enabled calorie tracking  
📊 Nutrition calculation and tracking
🏋️ 5 exercises with rep counting

**Popular Questions:**
- "How many calories should I eat?"
- "What's a good workout routine?"
- "How do I track my meals?"

Try our voice calorie tracker or ask me about specific exercises. I'm here to help you reach your fitness goals!"""
    
    def get_user_profile_summary(self):
        """Get user profile summary for display"""
        if not self.user_data:
            return "Profile not set up"
        
        return f"Weight: {self.user_data.get('weight', 'N/A')}kg | Goal: {self.user_data.get('primary_goal', 'N/A')}"
