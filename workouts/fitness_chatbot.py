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
        welcome_message = """
🏋️ Welcome to FitPro AI Trainer! 🏋️

I'm here to help you achieve your fitness goals! Let me learn about you first.

To create your personalized fitness plan, I need to know:
• Your height and weight
• Your fitness goals (weight loss, muscle gain, strength, etc.)
• Your current fitness level
• Any limitations or preferences

Let's start! What's your height and weight?
"""
        return welcome_message
    
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
    
    def generate_ollama_response(self, prompt: str) -> str:
        """Generate response using local Ollama LLM"""
        try:
            # Ollama API endpoint
            url = "http://localhost:11434/api/generate"
            
            # Create fitness-focused prompt
            system_prompt = """You are a professional fitness trainer and nutritionist AI assistant. 
            Provide helpful, accurate, and motivating fitness advice. Keep responses concise but informative.
            Focus on practical workout routines, proper form, nutrition guidance, and goal-specific recommendations.
            Always prioritize safety and encourage proper form over heavy weights."""
            
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            payload = {
                "model": "llama3.2",  # You can change this to any model you have installed
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 500
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
        """Fallback response when LLM is not available"""
        message_lower = message.lower()
        
        # Basic keyword matching for common fitness questions
        if any(word in message_lower for word in ['workout', 'exercise', 'routine']):
            return """Here's a basic full-body workout routine:
            
**Beginner Routine (3x/week):**
• Squats: 3 sets of 8-12 reps
• Push-ups: 3 sets of 5-15 reps
• Pull-ups/Assisted pull-ups: 3 sets of 3-8 reps
• Plank: 3 sets of 30-60 seconds
• Walking/Light cardio: 20-30 minutes

Focus on proper form and gradually increase intensity!"""

        elif any(word in message_lower for word in ['diet', 'nutrition', 'eat']):
            return """Basic nutrition guidelines:
            
**For general fitness:**
• Eat whole foods: lean proteins, vegetables, fruits, whole grains
• Stay hydrated: 8+ glasses of water daily
• Protein: 0.8-1.2g per kg body weight
• Eat balanced meals every 3-4 hours
• Limit processed foods and added sugars

Adjust portions based on your specific goals!"""

        else:
            return """I'm here to help with your fitness journey! You can ask me about:
            
• Workout routines and exercises
• Nutrition and diet advice  
• Form and technique tips
• Goal-specific training plans
• Recovery and rest recommendations

What would you like to know about fitness?"""
    
    def process_message(self, message: str, user_profile: Optional[Dict] = None) -> str:
        """Process user message and generate appropriate response"""
        self.conversation_history.append({"user": message, "timestamp": datetime.now()})
        
        # Extract measurements if present
        measurements = self.extract_measurements(message)
        if measurements:
            self.user_data.update(measurements)
            
            response = f"Great! I've noted:\n"
            if 'height' in measurements:
                response += f"• Height: {measurements['height']} cm\n"
            if 'weight' in measurements:
                response += f"• Weight: {measurements['weight']} kg\n"
                
            # Calculate BMI if we have both measurements
            if 'height' in self.user_data and 'weight' in self.user_data:
                bmi = self.user_data['weight'] / ((self.user_data['height']/100) ** 2)
                bmi_category = self.get_bmi_category(bmi)
                response += f"• BMI: {bmi:.1f} ({bmi_category})\n"
            
            response += "\nNow, what are your fitness goals? (e.g., lose weight, build muscle, get stronger, improve endurance)"
            return response
        
        # Extract goals if present
        goals = self.extract_goals(message)
        if goals:
            self.user_data['goals'] = goals
            response = f"Excellent! Your goals: {', '.join(goals)}\n\n"
            
            # Generate personalized advice using LLM
            context = self.build_context_for_llm()
            llm_prompt = f"""Based on this user profile: {context}
            
            Please provide:
            1. A personalized workout recommendation
            2. Basic nutrition advice
            3. Tips for achieving their goals
            
            User message: {message}"""
            
            llm_response = self.generate_ollama_response(llm_prompt)
            return response + llm_response
        
        # General fitness query - use LLM
        context = self.build_context_for_llm()
        llm_prompt = f"""User profile: {context}
        
        User question: {message}
        
        Please provide helpful fitness advice based on their profile and question."""
        
        return self.generate_ollama_response(llm_prompt)
    
    def build_context_for_llm(self) -> str:
        """Build context string for LLM from user data"""
        context_parts = []
        
        if 'height' in self.user_data and 'weight' in self.user_data:
            bmi = self.user_data['weight'] / ((self.user_data['height']/100) ** 2)
            context_parts.append(f"Height: {self.user_data['height']}cm, Weight: {self.user_data['weight']}kg, BMI: {bmi:.1f}")
        
        if 'goals' in self.user_data:
            context_parts.append(f"Goals: {', '.join(self.user_data['goals'])}")
        
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
