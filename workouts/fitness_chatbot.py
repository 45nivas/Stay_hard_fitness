"""
Fitness Chatbot powered by Ollama (qwen2.5:3b) for personalized fitness guidance.
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
        return """**OS Architect - Senior Fitness Trainer**

I specialize strictly in biomechanics, fitness programming, and clinical nutrition. I do not process queries outside these domains.

**Available Protocol Services:**
- Comprehensive transformation planning
- Periodized workout programming
- Advanced macro and caloric targeting
- Biomechanical form analysis

**How to interact:**
- "Design a 12-week transformation protocol"
- "What is the optimal protein target for hypertrophy?"

Please provide your fitness directive."""
    
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

        prompt = f"""[SYSTEM] You are OS Architect, a highly professional, senior fitness and nutrition trainer. You must maintain a clinical, authoritative, and professional tone at all times. Do not use excessive emojis or casual greetings.
You MUST only answer fitness, workout, exercise, and nutrition questions.
If the user asks anything unrelated to fitness, respond ONLY with: "I specialize strictly in fitness, biomechanics, and nutrition."

CRITICAL INSTRUCTION: If the user asks for a workout plan, a transformation program, or personal coaching, YOU MUST NOT generate the plan immediately. Instead, you MUST ask the user to provide the following details first so you can account for allergies, diet, and medical history carefully:
1. Age & Gender
2. Phone Number (WhatsApp)
3. City & Country
4. Height (cm)
5. Weight (kg)
6. Medical issues or injuries (if any) (If none, type "None")
7. Fitness level (Beginner / Intermediate / Advanced)
8. Diet type (Veg / Non-Veg / Eggitarian)
9. Fitness goal (Fat loss / Muscle gain / Both / Other)
10. Training hours per week (1-2 hrs / 3–5 hrs / 6+ hrs)
11. Occupation
12. Monthly budget for coaching

[USER PROFILE] {user_profile_data}
[USER MESSAGE] {message}

[INSTRUCTIONS] Give highly specific, professional advice. If they request a plan, output the 12-point questionnaire first. Keep responses structured and concise."""

        return self._query_llm(prompt)

    def generate_general_response(self, message, user_profile_data):
        """Generate general fitness responses."""

        prompt = f"""[SYSTEM] You are OS Architect, a strict, professional senior fitness and nutrition trainer.
You MUST maintain a highly professional, authoritative tone without excessive emojis.
You MUST only answer questions about fitness, workouts, nutrition, health, and exercise.

CRITICAL INSTRUCTION: If the user asks for a workout plan, a transformation program, or personal coaching, YOU MUST NOT generate the plan immediately. Instead, you MUST ask the user to provide the following details first so you can account for allergies, diet, and medical history carefully:
1. Age & Gender
2. Phone Number (WhatsApp)
3. City & Country
4. Height (cm) & Weight (kg)
5. Medical issues or injuries (if any)
6. Fitness level (Beginner / Intermediate / Advanced)
7. Diet type (Veg / Non-Veg / Eggitarian)
8. Fitness goal (Fat loss / Muscle gain)
9. Training hours per week (1-2 / 3–5 / 6+)
10. Occupation & Monthly budget for coaching

[USER PROFILE] {user_profile_data}
[USER MESSAGE] {message}

[INSTRUCTIONS] Be strictly professional and highly knowledgeable. Reference clinical biomechanics when helpful. If they request a plan, output the questionnaire first."""

        return self._query_llm(prompt)
    
    def _query_llm(self, prompt):
        """Query LLM with hybrid routing: Gemini 1.5 Flash → qwen2.5:3b → offline responses."""
        # 1. Primary: Gemini 1.5 Flash Cloud
        result = self._query_gemini_1_5(prompt)
        if result:
            return result
            
        # 2. Fallbacks: Local Ollama and Offline
        return self._query_ollama(prompt)

    def _query_gemini_1_5(self, prompt):
        """Query Gemini 1.5 Flash API as primary high-performance LLM."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.info("GEMINI_API_KEY not found in environment — skipping Gemini cloud query")
            return None
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            logger.info("Querying Gemini 1.5 Flash (Cloud)...")
            response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = data['candidates'][0]['content']['parts'][0]['text'].strip()
                if result:
                    logger.info("Gemini 1.5 Flash cloud query successful")
                    return result
            else:
                logger.warning("Gemini 1.5 Flash returned status %s: %s", response.status_code, response.text)
        except Exception as e:
            logger.error("Gemini 1.5 Flash query error: %s", e)
            
        return None

    def _query_ollama(self, prompt):
        """Query Ollama LLM using qwen2.5:3b."""

        # Primary: qwen2.5:3b (user's local lightweight model)
        result = self._call_ollama_model("qwen2.5:3b", prompt, timeout=60)
        if result:
            return result

        # Fallback: Offline hardcoded responses
        logger.warning("Ollama model (qwen2.5:3b) unavailable — using offline responses")
        return self._get_offline_response(prompt)

    def _warm_up_model(self):
        """Pre-load the primary model into memory for instant responses."""
        try:
            payload = {
                "model": "qwen2.5:3b",
                "prompt": "hi",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1}
            }
            requests.post(self.ollama_url, json=payload, timeout=120)
            logger.info("qwen2.5:3b pre-warmed and loaded into memory")
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
        import re
        message_match = re.search(r'\[USER MESSAGE\] (.*?)(\n\[|\Z)', prompt, re.DOTALL)
        user_message = message_match.group(1).lower() if message_match else ""

        # Workout-related responses
        if any(word in user_message for word in ['workout', 'exercise', 'squat', 'pushup', 'curl']):
            return """**OS Architect - Biomechanics Subsystem Active**
LLM Telemetry offline. Loading local database protocols:

**Core Modalities:**
- Integrated pose detection for hypertrophy targeting
- Clinical form analysis (OpenCV)
- Neuromuscular rep counting

Deploy standard protocol: 3-4 working sets of 8-12 reps per movement. Ensure progressive overload."""

        # Nutrition-related responses
        elif any(word in user_message for word in ['nutrition', 'calories', 'diet', 'protein', 'eggs', 'eat']):
            return """**OS Architect - Nutrition Subsystem Active**
LLM Telemetry offline. Loading evidence-based macro targets:

**Clinical Targets:**
- Minimum 1.6-2.2g protein per kg
- Base hydration: 35ml water per kg
- Maintain 300-500 kcal surplus/deficit based on current cycle

*Use Voice Tracking modules to log ingestion events.*"""

        # Default fitness response
        else:
            return """**OS Architect Telemetry Warning**
Unable to connect to local LLM (Ollama). Running in offline procedural mode.

**Available System Architectures:**
- Real-time pose detection and biomechanics form correction
- Voice-enabled calorie and macro tracking
- Clinical nutrition calculation
- Periodized strength protocols

System ready. Awaiting your directive."""
    
    def get_user_profile_summary(self):
        """Get user profile summary for display"""
        if not self.user_data:
            return "Profile not set up"
        
        return f"Weight: {self.user_data.get('weight', 'N/A')}kg | Goal: {self.user_data.get('primary_goal', 'N/A')}"
