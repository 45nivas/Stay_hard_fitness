"""
Fitness Chatbot using local LLM for personalized fitness guidance
Handles user queries about workouts, nutrition, and fitness goals
"""

import json
import re
from typing import Dict, List, Optional
from datetime import datetime
import requests

class FitnessChatbot:
    def __init__(self):
        self.user_data = {}
        self.conversation_history = []
        self.fitness_knowledge = {
            'exercises': {
                'weight_loss': ['cardio', 'running', 'cycling', 'swimming', 'HIIT', 'burpees', 'jumping jacks'],
                'muscle_gain': ['weightlifting', 'squats', 'deadlifts', 'bench press', 'pull-ups', 'push-ups'],
                'strength': ['powerlifting', 'compound movements', 'progressive overload', 'heavy weights'],
                'endurance': ['running', 'cycling', 'rowing', 'long-distance training', 'steady-state cardio'],
                'flexibility': ['yoga', 'stretching', 'pilates', 'dynamic warm-ups', 'foam rolling']
            },
            'nutrition': {
                'weight_loss': 'Caloric deficit, high protein, moderate carbs, healthy fats',
                'muscle_gain': 'Caloric surplus, high protein (1.6-2.2g/kg), complex carbs, healthy fats',
                'maintenance': 'Balanced macros, adequate protein, whole foods focus'
            }
        }
    
    def initialize_conversation(self) -> str:
        """Start the fitness assessment conversation"""
        # No welcome message - straight to business
        return ""
    
    def extract_measurements(self, message: str) -> Dict:
        """Extract height and weight from user message"""
        measurements = {}
        
        # Height patterns (cm, feet/inches)
        height_patterns = [
            r'(\d+)\s*cm',
            r'(\d+)\s*centimeters?',
            r"(\d+)'\s*(\d+)\"?",  # 5'10"
            r"(\d+)\s*feet?\s*(\d+)\s*inches?",
            r'(\d+\.?\d*)\s*m(?:eters?)?'  # 1.75m
        ]
        
        # Weight patterns (kg, lbs)
        weight_patterns = [
            r'(\d+\.?\d*)\s*kg',
            r'(\d+\.?\d*)\s*kilograms?',
            r'(\d+\.?\d*)\s*lbs?',
            r'(\d+\.?\d*)\s*pounds?'
        ]
        
        # Extract height
        for pattern in height_patterns:
            match = re.search(pattern, message.lower())
            if match:
                if "'" in pattern or "feet" in pattern:
                    # Convert feet/inches to cm
                    feet = int(match.group(1))
                    inches = int(match.group(2)) if match.group(2) else 0
                    measurements['height'] = round((feet * 30.48) + (inches * 2.54), 1)
                elif 'm' in pattern and 'cm' not in pattern:
                    # Convert meters to cm
                    measurements['height'] = float(match.group(1)) * 100
                else:
                    measurements['height'] = float(match.group(1))
                break
        
        # Extract weight
        for pattern in weight_patterns:
            match = re.search(pattern, message.lower())
            if match:
                weight = float(match.group(1))
                if 'lbs' in pattern or 'pounds' in pattern:
                    # Convert lbs to kg
                    measurements['weight'] = round(weight * 0.453592, 1)
                else:
                    measurements['weight'] = weight
                break
        
        return measurements
    
    def extract_goals(self, message: str) -> List[str]:
        """Extract fitness goals from user message"""
        goals = []
        goal_keywords = {
            'weight_loss': ['lose weight', 'weight loss', 'fat loss', 'slim down', 'cut', 'cutting'],
            'muscle_gain': ['gain muscle', 'build muscle', 'muscle gain', 'bulk', 'bulking', 'mass'],
            'strength': ['get stronger', 'build strength', 'strength training', 'powerlifting'],
            'endurance': ['endurance', 'stamina', 'cardio fitness', 'marathon', 'running'],
            'toning': ['tone up', 'toning', 'definition', 'lean muscle'],
            'general_fitness': ['get fit', 'fitness', 'healthy', 'overall health']
        }
        
        message_lower = message.lower()
        for goal, keywords in goal_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                goals.append(goal)
        
        return goals
    
    def calculate_bulking_calories(self, height: float, weight: float, age: int, gender: str = "Male") -> Dict[str, int]:
        """Calculate exact bulking calories and macros"""
        # Calculate BMR using Mifflin-St Jeor Equation
        if gender.lower() == "male":
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
        
        # TDEE (assuming moderate activity for gym-goers)
        tdee = bmr * 1.6  # Moderate exercise 3-5 days/week
        
        # Bulking calories (TDEE + 300-500 surplus)
        bulking_calories = int(tdee + 400)  # Conservative bulking surplus
        
        # Macros for bulking
        protein = int(weight * 2.2)  # 2.2g per kg for bulking
        fat = int(bulking_calories * 0.25 / 9)  # 25% of calories from fat
        carbs = int((bulking_calories - (protein * 4) - (fat * 9)) / 4)  # Rest from carbs
        
        return {
            'bmr': int(bmr),
            'tdee': int(tdee),
            'bulking_calories': bulking_calories,
            'protein': protein,
            'carbs': carbs,
            'fat': fat
        }

    def get_direct_fitness_answer(self, message: str, context: str) -> str:
        """No more hardcoded responses - everything goes to LLM"""
        # Remove all hardcoded responses, let LLM handle everything
        return None

    def generate_ollama_response(self, prompt: str) -> str:
        """Generate response using local Ollama LLM - 100% AI powered"""
        try:
            # Ollama API endpoint
            url = "http://localhost:11434/api/generate"
            
            # Enhanced fitness expert with calculation knowledge
            system_prompt = """You are a knowledgeable fitness expert and personal trainer. Give natural, conversational answers with specific numbers when needed.

            KEY KNOWLEDGE:
            - BMR calculation: Men = (10 × weight_kg) + (6.25 × height_cm) - (5 × age) + 5
            - TDEE = BMR × 1.6 (moderate activity)  
            - Bulking calories = TDEE + 400 calories surplus
            - Protein for bulking = 2.2g per kg body weight
            - Carbs = remaining calories after protein and fat
            - Fat = 25% of total calories

            RESPONSE STYLE:
            - Be conversational but knowledgeable
            - Give specific numbers and dosages
            - Use bullet points when helpful
            - No fluff or motivation talk
            - Act like a smart gym buddy who knows the science
            
            EXAMPLES:
            "Creatine timing?" → "Timing doesn't really matter bro. 5g daily, mix it with water or your protein shake. I take mine post-workout but pre-workout works too."
            "How many calories?" → Calculate using the formulas above and give the exact numbers with breakdown.
            """
            
            full_prompt = f"{system_prompt}\n\nUser Question: {prompt}\n\nAnswer:"
            
            payload = {
                "model": "llama3.2",
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Balance between consistency and naturalness
                    "max_tokens": 400,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Sorry, I could not generate a response.')
            else:
                return self.fallback_response(prompt)
                
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return self.fallback_response(prompt)
    
    def fallback_response(self, message: str) -> str:
        """Fallback response when LLM is not available - direct answers only"""
        message_lower = message.lower()
        
        # Basic keyword matching for common fitness questions
        if any(word in message_lower for word in ['workout', 'exercise', 'routine']):
            return """**BASIC BULKING WORKOUT:**
            
**Day 1,3,5:**
• Squats: 4x8-10
• Bench Press: 4x8-10  
• Bent Rows: 4x8-10
• Overhead Press: 3x8-10
• Deadlifts: 3x5 (Day 1 only)

**Day 2,4,6:**
• Pull-ups: 4x6-12
• Dips: 4x8-12
• Barbell Curls: 3x10-12
• Close-Grip Bench: 3x10-12

Rest 48-72 hours between sessions."""

        elif any(word in message_lower for word in ['calorie', 'diet', 'nutrition', 'eat']):
            return """**BULKING CALORIES (General):**
            
For 173cm, 21 years old:
• **Daily: 2800-3200 calories**
• **Protein: 140-160g**
• **Carbs: 350-400g** 
• **Fat: 90-110g**

**BULK FOODS:**
• Rice, oats, pasta (400g cooked daily)
• Chicken breast (200g daily)
• Eggs (3-4 whole daily)
• Nuts/peanut butter (30g daily)
• Whole milk (500ml daily)

Eat every 3 hours."""

        else:
            return """**AVAILABLE HELP:**
            
• Bulking calories & macros
• Workout routines with reps
• Specific exercise form
• Meal timing and foods
• Supplement timing

Ask specific questions like:
"How many calories for bulking?"
"What workout for chest growth?"
"When to take protein?" """
    
    def process_message(self, message: str, user_profile: Optional[Dict] = None) -> str:
        """Process user message and generate appropriate response"""
        self.conversation_history.append({"user": message, "timestamp": datetime.now()})
        
        # If user profile is provided, use it to override/update stored data
        if user_profile:
            # Update user data with profile information
            self.user_data.update({
                'height': user_profile.get('height'),
                'weight': user_profile.get('weight'),
                'age': user_profile.get('age'),
                'gender': user_profile.get('gender'),
                'fitness_level': user_profile.get('fitness_level'),
                'goals': [user_profile.get('primary_goal')] if user_profile.get('primary_goal') else self.user_data.get('goals', []),
                'primary_goal_code': user_profile.get('primary_goal_code'),
                'injuries_or_limitations': user_profile.get('injuries_or_limitations'),
                'available_time': user_profile.get('available_time'),
                'weak_muscles': user_profile.get('weak_muscles', []),
                'equipment_available': user_profile.get('equipment_available', []),
                'calories_per_day': user_profile.get('calories_per_day'),
                'bmi': user_profile.get('bmi'),
                'bmi_category': user_profile.get('bmi_category')
            })
        
        # Extract measurements if present in message
        measurements = self.extract_measurements(message)
        if measurements:
            self.user_data.update(measurements)
            # Skip fluff, continue to process the actual question
        
        # Extract goals if present
        goals = self.extract_goals(message)
        if goals:
            self.user_data['goals'] = goals
            # Skip fluff, continue to process the actual question
        
        # EVERYTHING goes to LLM now - no hardcoded responses
        context = self.build_context_for_llm()
        
        # Enhanced LLM prompt with user's complete profile data
        llm_prompt = f"""User Profile: {context}

User Question: {message}

Give a natural, conversational answer. If they ask about calories, calculate using:
- BMR: (10 × {self.user_data.get('weight', 75)}) + (6.25 × {self.user_data.get('height', 173)}) - (5 × {self.user_data.get('age', 21)}) + 5
- TDEE: BMR × 1.6 
- Bulking calories: TDEE + 400
- Protein: {self.user_data.get('weight', 75)} kg × 2.2g = {int(self.user_data.get('weight', 75) * 2.2)}g

Answer like a knowledgeable gym buddy, not a robot."""
        
        return self.generate_ollama_response(llm_prompt)
    
    def build_context_for_llm(self) -> str:
        """Build context string for LLM from user data"""
        context_parts = []
        
        # Basic measurements
        if 'height' in self.user_data and 'weight' in self.user_data:
            bmi = self.user_data.get('bmi') or (self.user_data['weight'] / ((self.user_data['height']/100) ** 2))
            bmi_category = self.user_data.get('bmi_category') or self.get_bmi_category(bmi)
            context_parts.append(f"Height: {self.user_data['height']}cm, Weight: {self.user_data['weight']}kg, BMI: {bmi:.1f} ({bmi_category})")
        
        # Demographics
        if 'age' in self.user_data:
            context_parts.append(f"Age: {self.user_data['age']}")
        if 'gender' in self.user_data:
            context_parts.append(f"Gender: {self.user_data['gender']}")
        
        # Fitness info
        if 'fitness_level' in self.user_data:
            context_parts.append(f"Fitness Level: {self.user_data['fitness_level']}")
        if 'goals' in self.user_data:
            context_parts.append(f"Primary Goal: {', '.join(self.user_data['goals'])}")
        elif 'primary_goal_code' in self.user_data:
            # Map goal codes to readable names
            goal_mapping = {
                'weight_loss': 'Weight Loss',
                'muscle_gain': 'Muscle Gain', 
                'bulking': 'Bulking',
                'cutting': 'Cutting',
                'strength': 'Build Strength',
                'endurance': 'Improve Endurance',
                'toning': 'Toning & Definition',
                'general_fitness': 'General Fitness',
                'maintaining': 'Maintaining'
            }
            goal_name = goal_mapping.get(self.user_data['primary_goal_code'], self.user_data['primary_goal_code'])
            context_parts.append(f"Primary Goal: {goal_name}")
        
        # Additional details
        if 'available_time' in self.user_data:
            context_parts.append(f"Available workout time: {self.user_data['available_time']} minutes")
        if 'weak_muscles' in self.user_data and self.user_data['weak_muscles']:
            weak_muscles = [m.strip() for m in self.user_data['weak_muscles'] if m.strip()]
            if weak_muscles:
                context_parts.append(f"Weak muscle groups: {', '.join(weak_muscles)}")
        if 'equipment_available' in self.user_data and self.user_data['equipment_available']:
            equipment = [e.strip() for e in self.user_data['equipment_available'] if e.strip()]
            if equipment:
                context_parts.append(f"Available equipment: {', '.join(equipment)}")
        if 'injuries_or_limitations' in self.user_data and self.user_data['injuries_or_limitations']:
            context_parts.append(f"Injuries/Limitations: {self.user_data['injuries_or_limitations']}")
        if 'calories_per_day' in self.user_data and self.user_data['calories_per_day']:
            context_parts.append(f"Daily calorie intake: {self.user_data['calories_per_day']} calories")
        
        return " | ".join(context_parts) if context_parts else "No profile data available"

    def get_bmi_category(self, bmi: float) -> str:
        """Get BMI category"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def get_user_profile_summary(self) -> Dict:
        """Get summary of collected user data"""
        summary = self.user_data.copy()
        
        if 'height' in self.user_data and 'weight' in self.user_data:
            bmi = self.user_data['weight'] / ((self.user_data['height']/100) ** 2)
            summary['bmi'] = round(bmi, 1)
            summary['bmi_category'] = self.get_bmi_category(bmi)
        
        return summary
