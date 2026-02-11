"""
Fitness Chatbot using Ollama for personalized fitness guidance
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
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        
        # Test Ollama connection
        self.ollama_available = self._test_ollama_connection()
        if not self.ollama_available:
            print("WARNING: Ollama not available. Make sure Ollama is running and llama3.2 model is installed.")
    
    def _test_ollama_connection(self):
        """Test if Ollama is available using lightweight tags endpoint"""
        try:
            # Use /api/tags instead of generating a response - much faster
            base_url = self.ollama_url.replace('/api/generate', '')
            response = requests.get(
                f"{base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                # Check if our model is available
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                if self.ollama_model.split(':')[0] in model_names:
                    return True
                else:
                    print(f"WARNING: Model '{self.ollama_model}' not found. Available: {model_names}")
                    return False
            return False
        except:
            return False
    
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
        # Specific calorie/macro calculation requests → hardcoded calculator
        elif any(keyword in message_lower for keyword in ['daily calories', 'how many calories', 'nutrition plan', 'macros', 'diet plan', 'bmr', 'tdee', 'meal plan', 'calorie goal', 'calorie target']):
            return self.calculate_daily_nutrition(user_profile_data)
        
        # General food/nutrition/diet questions → Ollama for smart answers
        elif any(keyword in message_lower for keyword in ['protein', 'carbs', 'food', 'foods', 'eat', 'nutrition', 'diet', 'calorie', 'water', 'hydration', 'drink', 'snack', 'meal', 'recipe', 'supplement', 'creatine', 'whey', 'vitamin']):
            return self.generate_nutrition_response(message, user_profile_data)
        
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
        """Generate workout responses with MediaPipe pose detection context and today's progress"""
        
        # Build today's workout summary
        today_summary = "No workouts completed today yet"
        if user_profile_data and user_profile_data.get('today_activity', {}).get('total_reps', 0) > 0:
            activity = user_profile_data['today_activity']
            today_summary = f"Today: {activity['total_reps']} total reps, {activity['total_calories_burned']} calories burned"
            if activity.get('by_exercise'):
                exercises = ", ".join([f"{ex}: {data['reps']} reps" for ex, data in activity['by_exercise'].items()])
                today_summary += f" | Exercises: {exercises}"
        
        prompt = f"""You are a fitness trainer for a gym app with OpenCV pose detection.
        
User message: "{message}"
User profile: {user_profile_data}
Today's Progress: {today_summary}

Available workouts with real-time pose correction:
- Squats (MediaPipe landmark tracking)
- Push-ups (Form analysis with OpenCV)  
- Bicep Curls (Rep counting)
- Hammer Curls (Left/right arm tracking)
- Side Raises (Shoulder form check)

Provide specific, actionable fitness advice. Reference today's completed workouts if any.
Encourage continued progress and suggest complementary exercises.
Keep responses under 200 words and gym-focused."""

        return self._query_ollama(prompt)
    
    def generate_nutrition_response(self, message, user_profile_data):
        """Generate nutrition/food responses using Ollama for smart, contextual answers"""
        
        profile_context = "No profile set up"
        if user_profile_data:
            weight = user_profile_data.get('weight', 'unknown')
            goal = user_profile_data.get('primary_goal_code', 'general fitness')
            profile_context = f"Weight: {weight}kg, Goal: {goal.replace('_', ' ')}"
        
        prompt = f"""You are a knowledgeable fitness nutrition coach.

User message: "{message}"
User profile: {profile_context}

Answer the user's specific question about food, nutrition, or diet directly.
If they ask about specific foods (e.g. Indian foods, high protein foods), give a detailed list with protein/calorie values per serving.
If they ask about water intake, give personalized advice based on their weight.
Be specific, practical, and helpful. Use emojis for readability.
Keep responses under 200 words."""

        return self._query_ollama(prompt)
    
    def generate_general_response(self, message, user_profile_data):
        """Generate general fitness responses with today's activity context"""
        
        # Build enhanced context with today's data
        context_parts = []
        if user_profile_data:
            # Add today's activity summary if available
            today_activity = user_profile_data.get('today_activity', {})
            today_nutrition = user_profile_data.get('today_nutrition', {})
            
            if today_activity.get('total_reps', 0) > 0 or today_activity.get('total_calories_burned', 0) > 0:
                context_parts.append(f"Today's workout: {today_activity['total_reps']} total reps, {today_activity['total_calories_burned']} calories burned")
                if today_activity.get('by_exercise'):
                    exercises = ", ".join([f"{ex}: {data['reps']} reps" for ex, data in today_activity['by_exercise'].items()])
                    context_parts.append(f"Exercises completed: {exercises}")
            
            if today_nutrition.get('total_calories', 0) > 0:
                context_parts.append(f"Today's nutrition: {today_nutrition['total_calories']} calories, {today_nutrition['total_protein']}g protein")
        
        today_context = " | ".join(context_parts) if context_parts else "No activity tracked today"
        
        prompt = f"""You are an AI fitness trainer for a modern gym app.
        
User message: "{message}"
User profile: {user_profile_data}
Today's Progress: {today_context}

Features available:
- Real-time pose correction with MediaPipe
- Voice calorie tracking  
- Rep counting for 5 exercises
- One rep max calculator

Be encouraging, knowledgeable, and reference today's progress when relevant.
If user has completed workouts today, congratulate them and provide insights.
Keep responses under 150 words."""

        return self._query_ollama(prompt)
    
    def _query_ollama(self, prompt):
        """Query Ollama local LLM for fitness responses"""
        if not self.ollama_available:
            return self._get_offline_response(prompt)
            
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500,
                    "num_ctx": 2048
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60  # Allow time for model loading on first query
            )
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('response', '').strip()
                if result:
                    return result
                else:
                    return self._get_offline_response(prompt)
            else:
                print(f"Ollama API error: {response.status_code}")
                return self._get_offline_response(prompt)
                
        except requests.exceptions.ConnectionError:
            print("Ollama not running - using offline responses")
            return self._get_offline_response(prompt)
        except requests.exceptions.Timeout:
            print("Ollama request timed out - using offline responses")
            return self._get_offline_response(prompt)
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._get_offline_response(prompt)
    
    def _get_offline_response(self, prompt):
        """Provide built-in responses for demo when API is unavailable"""
        prompt_lower = prompt.lower()
        
        # Extract just the user message from the prompt to avoid matching keywords
        # in the prompt template itself (e.g. "You are an AI fitness trainer")
        user_msg = ""
        msg_match = re.search(r'user message:\s*"(.+?)"', prompt_lower)
        if msg_match:
            user_msg = msg_match.group(1)
        
        # Check if this is a non-fitness question being processed - use word boundaries
        if any(re.search(r'\b' + word + r'\b', user_msg) for word in ['language model', 'llm', 'ai', 'technology', 'computer']):
            # For non-fitness questions, use the redirect method
            if re.search(r'\b(language model|llm)\b', user_msg):
                return self.handle_non_fitness_with_redirect("what is large language model")
            elif re.search(r'\b(technology|computer)\b', user_msg):
                return self.handle_non_fitness_with_redirect("technology question")
            else:
                # Handle general AI questions or fallback
                return self.handle_non_fitness_with_redirect("ai question")
        
        # Workout-related responses
        elif any(word in user_msg for word in ['workout', 'exercise', 'squat', 'pushup', 'curl']):
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
        elif any(word in user_msg for word in ['nutrition', 'calories', 'diet', 'protein', 'food', 'eat', 'meal']):
            return """🍎 **Nutrition Guide for Fitness Enthusiasts!**

🥩 **High Protein Foods:**
- Chicken breast: 31g protein per 100g
- Eggs: 13g protein per 2 eggs
- Paneer: 18g protein per 100g
- Greek yogurt/Curd: 10g protein per 100g
- Dal/Lentils: 9g protein per 100g (cooked)
- Chickpeas (Chana): 19g protein per 100g
- Soya chunks: 52g protein per 100g
- Fish/Tuna: 26g protein per 100g
- Whey protein: 24g per scoop

📊 **General Guidelines:**
- Eat 1.6-2.2g protein per kg body weight
- Stay hydrated: 35ml water per kg body weight
- Eat protein within 30 mins post-workout
- Split protein across 4-5 meals for better absorption

📱 **Use Our Voice Tracker:**
Say your meals aloud and let AI track your calories automatically!

💡 **Pro Tip:** Set up your profile for a personalized nutrition plan with exact macro targets!"""

        # Water/hydration responses
        elif any(word in user_msg for word in ['water', 'hydration', 'hydrate', 'drink', 'thirst']):
            return """💧 **Hydration Guide for Active People:**

**General Rule:** Drink **35ml of water per kg of body weight** daily.

📊 **Quick Reference:**
- 60kg → ~2.1 liters/day
- 70kg → ~2.5 liters/day
- 80kg → ~2.8 liters/day
- 90kg → ~3.2 liters/day
- 100kg → ~3.5 liters/day

🏋️ **During Workouts:**
- Drink 200-300ml **before** exercise
- Sip 150-200ml every **15-20 minutes** during exercise
- Drink 500ml **after** your workout

⚡ **Signs You Need More Water:**
- Dark yellow urine
- Feeling fatigued or dizzy
- Dry mouth or headaches
- Decreased workout performance

💡 **Pro Tips:**
- Start your day with 500ml of water
- Keep a water bottle during workouts
- Set up your profile for a personalized recommendation!
- Add electrolytes for intense sessions (45+ mins)"""

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

    # ── Streaming support ──────────────────────────────────────────────
    def _build_prompt(self, message, user_profile_data=None):
        """Build the prompt for Ollama based on message routing (same logic as process_message)"""
        message_lower = message.lower()

        # Sleep and lifestyle
        if any(kw in message_lower for kw in ['sleep', 'rest', 'recovery', 'tired', 'time management', 'busy', 'schedule']):
            return self._general_prompt(message, user_profile_data)
        # Fat loss
        elif any(kw in message_lower for kw in ['lose fat', 'fat loss', 'cutting', 'lose weight']):
            return None  # hardcoded response, no streaming needed
        # Calorie calculator
        elif any(kw in message_lower for kw in ['daily calories', 'how many calories', 'nutrition plan', 'macros', 'diet plan', 'bmr', 'tdee', 'meal plan', 'calorie goal', 'calorie target']):
            return None
        # Nutrition / food questions
        elif any(kw in message_lower for kw in ['protein', 'carbs', 'food', 'foods', 'eat', 'nutrition', 'diet', 'calorie', 'water', 'hydration', 'drink', 'snack', 'meal', 'recipe', 'supplement', 'creatine', 'whey', 'vitamin']):
            return self._nutrition_prompt(message, user_profile_data)
        # Workouts
        elif any(kw in message_lower for kw in ['workout', 'exercise', 'squat', 'pushup', 'curl', 'rep']):
            return self._workout_prompt(message, user_profile_data)
        # Non-fitness redirect
        elif any(re.search(r'\b' + kw + r'\b', message_lower) for kw in ['language model', 'llm', 'ai', 'technology', 'computer', 'programming']):
            return None
        # General
        else:
            return self._general_prompt(message, user_profile_data)

    def _nutrition_prompt(self, message, user_profile_data):
        profile_context = "No profile set up"
        if user_profile_data:
            weight = user_profile_data.get('weight', 'unknown')
            goal = user_profile_data.get('primary_goal_code', 'general fitness')
            profile_context = f"Weight: {weight}kg, Goal: {goal.replace('_', ' ')}"
        return f"""You are a knowledgeable fitness nutrition coach.

User message: "{message}"
User profile: {profile_context}

Answer the user's specific question about food, nutrition, or diet directly.
If they ask about specific foods (e.g. Indian foods, high protein foods), give a detailed list with protein/calorie values per serving.
If they ask about water intake, give personalized advice based on their weight.
Be specific, practical, and helpful. Use emojis for readability.
Keep responses under 200 words."""

    def _workout_prompt(self, message, user_profile_data):
        today_summary = "No workouts completed today yet"
        if user_profile_data and user_profile_data.get('today_activity', {}).get('total_reps', 0) > 0:
            activity = user_profile_data['today_activity']
            today_summary = f"Today: {activity['total_reps']} total reps, {activity['total_calories_burned']} calories burned"
        return f"""You are a fitness trainer for a gym app with OpenCV pose detection.

User message: "{message}"
User profile: {user_profile_data}
Today's Progress: {today_summary}

Available workouts with real-time pose correction:
- Squats, Push-ups, Bicep Curls, Hammer Curls, Side Raises

Provide specific, actionable fitness advice. Keep responses under 200 words and gym-focused."""

    def _general_prompt(self, message, user_profile_data):
        context_parts = []
        if user_profile_data:
            today_activity = user_profile_data.get('today_activity', {})
            today_nutrition = user_profile_data.get('today_nutrition', {})
            if today_activity.get('total_reps', 0) > 0:
                context_parts.append(f"Today's workout: {today_activity['total_reps']} reps, {today_activity['total_calories_burned']} cal burned")
            if today_nutrition.get('total_calories', 0) > 0:
                context_parts.append(f"Nutrition: {today_nutrition['total_calories']} cal, {today_nutrition['total_protein']}g protein")
        today_context = " | ".join(context_parts) if context_parts else "No activity tracked today"
        return f"""You are an AI fitness trainer for a modern gym app.

User message: "{message}"
User profile: {user_profile_data}
Today's Progress: {today_context}

Be encouraging, knowledgeable, and reference today's progress when relevant.
Keep responses under 150 words."""

    def stream_message(self, message, user_profile_data=None):
        """Stream response from Ollama token-by-token. Yields text chunks.
        Falls back to yielding the full non-streamed response at once."""
        prompt = self._build_prompt(message, user_profile_data)
        if prompt is None:
            # Hardcoded response – yield all at once
            yield self.process_message(message, user_profile_data)
            return

        if not self.ollama_available:
            yield self._get_offline_response(prompt)
            return

        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 2048
                }
            }
            with requests.post(self.ollama_url, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    yield self._get_offline_response(prompt)
                    return
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get('response', '')
                            if token:
                                yield token
                            if data.get('done', False):
                                return
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Ollama stream error: {e}")
            yield self._get_offline_response(prompt)
