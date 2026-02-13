"""
Fitness Chatbot powered by Ollama (llama3.2) for personalized fitness guidance.
Handles user queries about workouts, nutrition, and fitness goals.
Off-topic questions are politely redirected to fitness topics.
"""

import re
import logging
import requests
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Fitness-related keywords used to determine if a message is on-topic
FITNESS_KEYWORDS = [
    # Exercises
    'workout', 'exercise', 'squat', 'pushup', 'push-up', 'push up',
    'curl', 'deadlift', 'bench press', 'overhead press', 'plank', 'lunge',
    'pull-up', 'pull up', 'chin up', 'dip', 'row', 'crunch', 'burpee',
    'kettlebell', 'dumbbell', 'barbell', 'resistance band',
    # Muscle groups
    'muscle', 'bicep', 'tricep', 'chest', 'quadricep', 'hamstring',
    'glute', 'deltoid', 'lat', 'trapezius', 'forearm', 'calf',
    'abs', 'core', 'shoulder press', 'leg press', 'leg day', 'arm day',
    # Nutrition (specific)
    'protein', 'carb', 'calorie', 'macro', 'micronutrient',
    'diet', 'nutrition', 'bulking', 'cutting', 'lean bulk',
    'supplement', 'creatine', 'whey', 'bcaa', 'pre-workout', 'post-workout',
    'intermittent fasting', 'meal prep', 'cheat meal', 'caloric deficit',
    'caloric surplus', 'protein shake', 'mass gainer',
    # Fitness concepts
    'gym', 'fitness', 'strength training', 'weight training', 'cardio',
    'hiit', 'warm up', 'cool down', 'stretching', 'flexibility', 'mobility',
    'endurance', 'stamina', 'hypertrophy', 'progressive overload',
    'one rep max', '1rm', 'pr', 'personal record', 'rep', 'reps',
    # Health & recovery
    'bmr', 'tdee', 'bmi', 'heart rate', 'vo2',
    'sleep', 'recovery', 'rest day', 'muscle soreness', 'doms',
    'injury', 'sprain', 'posture', 'spine', 'joint',
    'hydration', 'electrolyte',
    # Goals
    'lose weight', 'gain weight', 'fat loss', 'muscle gain', 'bulk up',
    'get fit', 'get lean', 'get strong', 'get ripped', 'get shredded',
    'weight loss', 'body fat', 'six pack',
    # App features
    'pose detection', 'rep count', 'calorie track', 'form check',
]


class FitnessChatbot:
    def __init__(self):
        self.user_data = {}
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        # Pre-warm the model on first init so it's loaded in memory
        self._warm_up_model()

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

    def _is_fitness_related(self, message):
        """Check if a message is related to fitness, health, or our app features."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in FITNESS_KEYWORDS)

    def _is_off_topic(self, message):
        """Detect clearly off-topic questions (celebrity, politics, general knowledge, etc.)"""
        message_lower = message.lower().strip()

        # Pattern: "who is X", "what is X" where X is not fitness-related
        who_what_pattern = re.match(r'^(who|what|where|when|why|how)\s+(is|are|was|were|do|does|did)\s+', message_lower)
        if who_what_pattern and not self._is_fitness_related(message):
            return True

        # Pattern: "tell me about X" where X is not fitness-related
        if re.match(r'^(tell me about|explain|describe|define)\s+', message_lower) and not self._is_fitness_related(message):
            return True

        # Explicit off-topic keywords
        off_topic_keywords = [
            'movie', 'film', 'actor', 'actress', 'celebrity', 'song', 'music',
            'politics', 'election', 'president', 'minister', 'cricket', 'football score',
            'recipe', 'cook', 'weather', 'news', 'stock market', 'crypto',
            'anime', 'manga', 'game of thrones', 'netflix', 'youtube',
            'programming', 'python code', 'javascript', 'html', 'css',
        ]
        if any(kw in message_lower for kw in off_topic_keywords) and not self._is_fitness_related(message):
            return True

        return False

    def process_message(self, message, user_profile_data=None):
        """Process user message with intelligent routing and fitness focus."""
        message_lower = message.lower()

        # 1. Catch off-topic questions FIRST ("who is alluarjun", "tell me about movies", etc.)
        if self._is_off_topic(message):
            return self._redirect_to_fitness()

        # 2. Tech/AI questions — answer briefly then redirect
        if any(re.search(r'\b' + kw + r'\b', message_lower) for kw in ['language model', 'llm', 'ai model', 'chatgpt', 'gpt']):
            return self.handle_non_fitness_with_redirect(message)

        # 3. Fat loss queries
        if any(kw in message_lower for kw in ['lose fat', 'fat loss', 'cutting', 'lose weight']):
            return self.provide_fat_loss_advice(message, user_profile_data)

        # 4. Specific food questions ("how many calories in an egg") → send to LLM
        food_question_patterns = [
            r'how many (calories|carbs|protein|fat) .*(in|does|do|has|have)',
            r'(calories|carbs|protein|fat) (in|of) \w+',
            r'is .* healthy',
            r'(should i eat|can i eat|is .* good for)',
        ]
        if any(re.search(pat, message_lower) for pat in food_question_patterns):
            return self.generate_general_response(message, user_profile_data)

        # 5. Personalized nutrition plan (my daily calories, my macros, etc.)
        if any(kw in message_lower for kw in [
            'my daily calories', 'my calorie', 'my macros', 'my nutrition',
            'nutrition plan', 'diet plan', 'bmr', 'tdee', 'meal plan',
            'what should i eat', 'my protein intake', 'my carb intake',
            'calculate my', 'how much should i eat', 'daily intake'
        ]):
            return self.calculate_daily_nutrition(user_profile_data)

        # 6. Workout queries
        if any(kw in message_lower for kw in ['workout', 'exercise', 'squat', 'pushup', 'curl', 'rep', 'deadlift', 'bench press']):
            return self.generate_workout_response(message, user_profile_data)

        # 6. Sleep / recovery / lifestyle
        if any(kw in message_lower for kw in ['sleep', 'rest', 'recovery', 'tired', 'schedule', 'time management']):
            return self.generate_general_response(message, user_profile_data)

        # 7. If it contains ANY fitness keyword, let the LLM handle it
        if self._is_fitness_related(message):
            return self.generate_general_response(message, user_profile_data)

        # 8. Nothing matched — redirect politely
        return self._redirect_to_fitness()

    def _redirect_to_fitness(self):
        """Polite redirect for off-topic questions."""
        return """🏋️ **I'm your AI Fitness Coach!**

I'm specialized in fitness, workouts, and nutrition — I can't help with other topics.

**Here's what I can do for you:**
- 💪 Personalized workout routines
- 🍎 Daily calorie & macro calculations
- 🎯 Exercise form tips with pose detection
- 📊 Fat loss / muscle gain plans

**Try asking:**
- "Give me a workout routine"
- "How many calories should I eat?"
- "What's my daily protein target?"
- "How do I do squats with proper form?"

Let's crush your fitness goals! 💪"""
    
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
        """Generate workout responses with MediaPipe pose detection context."""

        prompt = f"""[SYSTEM] You are a strict fitness-only trainer for a gym app with real-time pose detection.
You MUST only answer fitness, workout, exercise, and nutrition questions.
If the user asks anything unrelated to fitness (celebrities, movies, general knowledge, etc.), 
respond ONLY with: "I'm your fitness coach! Ask me about workouts, nutrition, or exercise form."
NEVER answer off-topic questions. NEVER make up personal details about the user.

[USER PROFILE] {user_profile_data}

[APP FEATURES]
- Squats with MediaPipe landmark tracking
- Push-ups with form analysis (OpenCV)
- Bicep Curls with rep counting
- Hammer Curls with left/right arm tracking
- Side Raises with shoulder form check

[USER MESSAGE] {message}

[INSTRUCTIONS] Give specific, actionable fitness advice. Mention pose detection features when relevant.
Keep response under 200 words. Use bullet points and emojis for readability."""

        return self._query_ollama(prompt)

    def generate_general_response(self, message, user_profile_data):
        """Generate general fitness responses."""

        prompt = f"""[SYSTEM] You are a strict fitness-only AI trainer for a modern gym app.
You MUST only answer questions about fitness, workouts, nutrition, health, and exercise.
If the user asks about anything else (celebrities, movies, tech, general knowledge, etc.),
respond ONLY with: "I'm your fitness coach! Ask me about workouts, nutrition, or exercise form."
NEVER answer off-topic questions. NEVER hallucinate or make up personal information.

[USER PROFILE] {user_profile_data}

[APP FEATURES]
- Real-time pose correction with MediaPipe
- Voice calorie tracking
- Rep counting for 5 exercises
- One rep max calculator

[USER MESSAGE] {message}

[INSTRUCTIONS] Be encouraging and knowledgeable. Reference app features when helpful.
Keep response under 150 words. Use emojis and formatting for readability."""

        return self._query_ollama(prompt)
    
    def _query_ollama(self, prompt):
        """Query Ollama LLM with fallback: mistral:7b → llama3 → offline responses."""

        # Model 1: mistral:7b (primary — best instruction-following, clean responses)
        result = self._call_ollama_model("mistral:7b", prompt, timeout=60)
        if result:
            return result

        # Model 2: llama3 8B (fallback — strong general model)
        logger.info("Primary model unavailable, falling back to llama3")
        result = self._call_ollama_model("llama3", prompt, timeout=120)
        if result:
            return result

        # Model 3: Offline hardcoded responses
        logger.warning("All Ollama models unavailable — using offline responses")
        return self._get_offline_response(prompt)

    def _warm_up_model(self):
        """Pre-load the primary model into memory for instant responses."""
        try:
            payload = {
                "model": "mistral:7b",
                "prompt": "hi",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1}
            }
            requests.post(self.ollama_url, json=payload, timeout=120)
            logger.info("mistral:7b pre-warmed and loaded into memory")
        except Exception:
            logger.info("Could not pre-warm model (Ollama may not be running)")

    def _call_ollama_model(self, model_name, prompt, timeout=15):
        """Call a specific Ollama model with given timeout. Returns None on failure."""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 300
                }
            }

            logger.info("Querying Ollama (%s)...", model_name)
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=timeout
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get('response', '').strip()
                if result:
                    logger.info("%s responded successfully", model_name)
                    return result
            else:
                logger.warning("%s returned status %s", model_name, response.status_code)

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not running — %s unavailable", model_name)
        except requests.exceptions.ReadTimeout:
            logger.warning("%s timed out after %ss", model_name, timeout)
        except Exception as e:
            logger.error("%s error: %s", model_name, e)

        return None
    
    def _get_offline_response(self, prompt):
        """Provide built-in responses when Ollama is unavailable (demo / offline mode)."""
        prompt_lower = prompt.lower()

        # Workout-related responses
        if any(word in prompt_lower for word in ['workout', 'exercise', 'squat', 'pushup', 'curl']):
            return """💪 **Great question about workouts!**

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

Start with 3 sets of 8-12 reps. Our pose detection will help you maintain perfect form! 💪"""

        # Nutrition-related responses
        elif any(word in prompt_lower for word in ['nutrition', 'calories', 'diet', 'protein']):
            return """🍎 **Nutrition is key to your fitness success!**

**General Guidelines:**
- Eat 1.6-2.2g protein per kg body weight
- Stay hydrated: 35ml water per kg body weight
- Eat within 30 mins post-workout

📱 **Use Our Voice Tracker:**
- Say "I ate 200g chicken breast" for instant tracking
- Real-time nutrition calculations
- Accurate macro breakdowns

**Pro Tip:** Use our calorie tracker to log meals easily — just speak into the microphone! 🎤"""

        # Default fitness response
        else:
            return """💪 **I'm your AI Fitness Coach!**

**What I can help with:**
🎯 Real-time pose detection for perfect form
🎤 Voice-enabled calorie tracking
📊 Nutrition calculation and tracking
🏋️ 5 exercises with rep counting

**Try asking:**
- "How many calories should I eat?"
- "Give me a workout routine"
- "How do I track my meals?"

Let's crush your fitness goals! 🔥"""
    
    def get_user_profile_summary(self):
        """Get user profile summary for display"""
        if not self.user_data:
            return "Profile not set up"
        
        return f"Weight: {self.user_data.get('weight', 'N/A')}kg | Goal: {self.user_data.get('primary_goal', 'N/A')}"
