"""
TASK: Integrate LangChain with Ollama in a Django gym app.

GOAL:
- Take user profile (age, gender, height, weight, fitness level, goal)
- Use LangChain with Ollama (e.g., llama3 or mistral) to generate:
    1. Personalized 7-day workout plan
    2. Macronutrient recommendations (calories, protein, etc.)
    3. Workout progression advice
- Response format should be JSON, easy to render in Django templates.

Steps to implement:
1. Connect to Ollama using LangChain's `ChatOllama` class
2. Create a LangChain prompt template using user data
3. Send prompt and get AI-generated plan
4. Return it in Django view as context

Use local model like `llama3` or `mistral` running on Ollama.

Add proper error handling if Ollama is not available.
"""

import json
import logging
from django.conf import settings

try:
    from langchain_ollama import ChatOllama
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as ChatOllama
        from langchain.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class WorkoutRecommendationEngine:
    def __init__(self):
        self.llm = None
        self.setup_ollama()
    
    def setup_ollama(self):
        """Connect to Ollama using LangChain"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available")
            return
            
        try:
            # Initialize Ollama with LangChain
            self.llm = ChatOllama(
                model="llama3",  # or "mistral"
                base_url="http://localhost:11434"
            )
            logger.info("Successfully connected to Ollama via LangChain")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            self.llm = None
    
    def generate_recommendations(self, user_profile):
        """Generate AI workout recommendations using LangChain + Ollama"""
        if not self.llm:
            return self._get_fallback_recommendation(user_profile)
        
        try:
            # Create LangChain prompt template
            prompt_template = PromptTemplate(
                input_variables=[
                    "name", "age", "gender", "height", "weight", 
                    "fitness_level", "goal", "equipment", "weak_muscles", "limitations", "workout_time"
                ],
                template=self._get_prompt_template()
            )
            
            # Create modern LangChain chain using pipe operator
            output_parser = StrOutputParser()
            chain = prompt_template | self.llm | output_parser
            
            # Prepare user data
            user_data = {
                "name": user_profile.user.first_name or user_profile.user.username,
                "age": user_profile.age,
                "gender": user_profile.get_gender_display(),
                "height": user_profile.height,
                "weight": user_profile.weight,
                "fitness_level": user_profile.get_fitness_level_display(),
                "goal": user_profile.get_primary_goal_display(),
                "equipment": user_profile.equipment_available or "Bodyweight only",
                "weak_muscles": user_profile.weak_muscles or "None specified",
                "limitations": user_profile.injuries_or_limitations or "None",
                "workout_time": user_profile.available_time
            }
            
            # Generate AI response using invoke
            response = chain.invoke(user_data)
            
            # Try to parse JSON response
            try:
                ai_plan = json.loads(response)
                return self._format_ai_response(ai_plan, user_profile)
            except json.JSONDecodeError:
                # If not JSON, return as text
                return self._format_text_response(response, user_profile)
                
        except Exception as e:
            logger.error(f"Error generating AI recommendations: {e}")
            return self._get_fallback_recommendation(user_profile)
    
    def _get_prompt_template(self):
        """LangChain prompt template for workout generation"""
        return """
You are a professional fitness expert and certified personal trainer. Create a personalized workout plan for the following user:

Name: {name}
Age: {age}
Gender: {gender}
Height: {height} cm
Weight: {weight} kg
Fitness level: {fitness_level}
Goal: {goal}
Workout time per day: {workout_time} minutes
Equipment Available: {equipment}
Weak Muscles: {weak_muscles}
Limitations/Injuries: {limitations}

Create a comprehensive, flexible workout plan based on their available time, goal, and experience level. 

IMPORTANT GUIDELINES:
- Design the optimal workout schedule (3-6 days per week based on goals and time)
- Include progressive overload principles
- Add appropriate rest days for recovery
- Provide short nutritional guidance
- Focus on their specific goal and fitness level

Format your response as JSON with this structure:

{{
    "user_stats": {{
        "bmi": "calculated BMI",
        "tdee": "calculated daily calories",
        "bmi_category": "Normal/Overweight/etc",
        "recommended_calories": "adjusted calories for goal"
    }},
    "workout_schedule": {{
        "frequency": "4 days per week",
        "workout_days": {{
            "Day 1": {{
                "workout_name": "Upper Body Strength",
                "type": "strength",
                "duration": "45-60 minutes",
                "exercises": [
                    {{
                        "name": "Push-ups",
                        "sets": 3,
                        "reps": "8-12",
                        "rest": "60 seconds",
                        "targets": ["chest", "triceps", "shoulders"],
                        "instructions": "Keep core tight, full range of motion"
                    }}
                ]
            }},
            "Day 2": {{"workout_name": "Rest Day", "type": "rest", "activity": "Light walking or stretching"}},
            "Day 3": {{...continue for optimal schedule...}}
        }}
    }},
    "nutrition": {{
        "daily_calories": "number based on TDEE and goal",
        "protein_grams": "number", 
        "carbs_grams": "number",
        "fat_grams": "number",
        "hydration": "daily water intake recommendation",
        "meal_timing": "pre/post workout nutrition advice"
    }},
    "progression": {{
        "weekly": "How to progress each week",
        "monthly": "Long-term progression strategy"
    }},
    "tips": {{
        "form": "Key form tips for their workouts",
        "motivation": "Motivational advice for their goal",
        "recovery": "Recovery and rest recommendations"
    }}
}}

Focus on their goal: {goal}. Make it practical and achievable for their {fitness_level} level.
Calculate BMI and TDEE accurately. Ensure proper rest and progressive overload.
"""
    
    def _format_ai_response(self, ai_plan, user_profile):
        """Format AI JSON response for Django templates"""
        # Calculate actual BMI and TDEE if not provided
        bmi = round(user_profile.weight / ((user_profile.height/100) ** 2), 1)
        
        # Basic TDEE calculation (Harris-Benedict)
        if user_profile.gender == 'M':
            bmr = 88.362 + (13.397 * user_profile.weight) + (4.799 * user_profile.height) - (5.677 * user_profile.age)
        else:
            bmr = 447.593 + (9.247 * user_profile.weight) + (3.098 * user_profile.height) - (4.330 * user_profile.age)
        
        activity_multipliers = {'beginner': 1.375, 'intermediate': 1.55, 'advanced': 1.725}
        tdee = round(bmr * activity_multipliers.get(user_profile.fitness_level, 1.55))
        
        # Ensure required fields exist
        if 'user_stats' not in ai_plan:
            ai_plan['user_stats'] = {}
        
        ai_plan['user_stats'].update({
            'bmi': bmi,
            'tdee': tdee,
            'bmi_category': self._get_bmi_category(bmi)
        })
        
        # Handle both new and old format
        if 'workout_schedule' in ai_plan and 'workout_days' in ai_plan['workout_schedule']:
            ai_plan['weekly_plan'] = ai_plan['workout_schedule']['workout_days']
        elif 'weekly_plan' not in ai_plan:
            ai_plan['weekly_plan'] = {}
        
        return ai_plan
    
    def _format_text_response(self, response, user_profile):
        """Format text response when JSON parsing fails"""
        bmi = round(user_profile.weight / ((user_profile.height/100) ** 2), 1)
        tdee = 2000  # Default estimate
        
        return {
            'user_stats': {
                'bmi': bmi,
                'tdee': tdee,
                'bmi_category': self._get_bmi_category(bmi)
            },
            'weekly_plan': {
                'Day 1': {'workout_name': 'AI Generated Plan', 'type': 'custom', 'description': response[:500]},
                'Day 2': {'workout_name': 'Rest Day', 'type': 'rest'},
                'Day 3': {'workout_name': 'Continue AI Plan', 'type': 'custom'},
                'Day 4': {'workout_name': 'Rest Day', 'type': 'rest'},
                'Day 5': {'workout_name': 'AI Strength Training', 'type': 'custom'},
                'Day 6': {'workout_name': 'Rest Day', 'type': 'rest'},
                'Day 7': {'workout_name': 'Rest Day', 'type': 'rest'}
            },
            'ai_response': response,
            'nutrition': {'daily_calories': tdee, 'advice': 'Follow AI recommendations above'},
            'general_advice': 'AI-generated personalized plan created successfully'
        }
    
    def _get_bmi_category(self, bmi):
        """Get BMI category"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def _get_fallback_recommendation(self, user_profile):
        """Fallback when Ollama/LangChain unavailable"""
        bmi = round(user_profile.weight / ((user_profile.height/100) ** 2), 1)
        
        return {
            'user_stats': {
                'bmi': bmi,
                'tdee': 2000,
                'bmi_category': self._get_bmi_category(bmi)
            },
            'weekly_plan': {
                'Day 1': {'workout_name': 'Full Body Workout', 'type': 'beginner'},
                'Day 2': {'workout_name': 'Rest Day', 'type': 'rest'},
                'Day 3': {'workout_name': 'Cardio Day', 'type': 'cardio'},
                'Day 4': {'workout_name': 'Rest Day', 'type': 'rest'},
                'Day 5': {'workout_name': 'Strength Training', 'type': 'strength'},
                'Day 6': {'workout_name': 'Rest Day', 'type': 'rest'},
                'Day 7': {'workout_name': 'Rest Day', 'type': 'rest'}
            },
            'nutrition': {
                'daily_calories': 2000,
                'advice': 'Eat balanced meals with adequate protein'
            },
            'general_advice': 'Ollama AI not available. Using basic workout template.',
            'fallback_used': True
        }
