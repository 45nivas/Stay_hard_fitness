"""
Fitness Chatbot using local LLM for personalized fitness guidance
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

    def process_message(self, message, user_profile_data=None):
        """Process user message with OpenCV/MediaPipe context"""
        message_lower = message.lower()
        
        # Nutrition queries
        if any(keyword in message_lower for keyword in ['calories', 'nutrition', 'macros', 'diet', 'bmr', 'tdee', 'eat']):
            return self.calculate_daily_nutrition(user_profile_data)
        
        # Workout queries with pose detection context
        elif any(keyword in message_lower for keyword in ['workout', 'exercise', 'squat', 'pushup', 'curl', 'rep']):
            return self.generate_workout_response(message, user_profile_data)
        
        # General fitness chat
        else:
            return self.generate_general_response(message, user_profile_data)
    
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
        """Query Ollama LLM for fitness responses"""
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 2048
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            response_json = response.json()
            return response_json.get("response", "I'm here to help with your fitness goals! Ask me about workouts or nutrition.")
            
        except Exception as e:
            return "💪 I'm experiencing technical difficulties, but I'm still here to support your fitness journey! Try asking about workouts or nutrition."
    
    def get_user_profile_summary(self):
        """Get user profile summary for display"""
        if not self.user_data:
            return "Profile not set up"
        
        return f"Weight: {self.user_data.get('weight', 'N/A')}kg | Goal: {self.user_data.get('primary_goal', 'N/A')}"
