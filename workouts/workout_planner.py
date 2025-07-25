"""
Advanced Workout Recommendation System
Generates personalized 7-day workout plans based on user profile and goals
"""

def calculate_bmi(height_cm, weight_kg):
    """Calculate BMI"""
    height_m = height_cm / 100
    return round(weight_kg / (height_m * height_m), 1)

def calculate_tdee(age, gender, height_cm, weight_kg, fitness_level):
    """Calculate Total Daily Energy Expenditure"""
    # Calculate BMR using Mifflin-St Jeor Equation
    if gender.lower() == 'm':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    # Activity factor based on fitness level
    activity_factors = {
        'beginner': 1.2,      # Sedentary
        'intermediate': 1.375, # Light exercise
        'advanced': 1.55      # Moderate exercise
    }
    
    return round(bmr * activity_factors.get(fitness_level, 1.2))

def get_exercise_database():
    """Return comprehensive exercise database with muscle groups and equipment"""
    return {
        # Current app exercises (with posture correction)
        'squats': {
            'name': 'Squats',
            'muscle_group': 'legs',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['quadriceps', 'glutes', 'hamstrings'],
            'has_posture_check': True
        },
        'push-ups': {
            'name': 'Push-ups',
            'muscle_group': 'chest',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['chest', 'triceps', 'shoulders'],
            'has_posture_check': True
        },
        'bicep_curls': {
            'name': 'Bicep Curls',
            'muscle_group': 'arms',
            'type': 'gym',
            'equipment': ['dumbbells'],
            'targets': ['biceps'],
            'has_posture_check': True
        },
        'hammer_curls': {
            'name': 'Hammer Curls',
            'muscle_group': 'arms',
            'type': 'gym',
            'equipment': ['dumbbells'],
            'targets': ['biceps', 'forearms'],
            'has_posture_check': True
        },
        'side_raises': {
            'name': 'Side Raises',
            'muscle_group': 'shoulders',
            'type': 'gym',
            'equipment': ['dumbbells'],
            'targets': ['side_delts'],
            'has_posture_check': True
        },
        
        # Additional exercises for comprehensive plans
        'incline_push_ups': {
            'name': 'Incline Push-ups',
            'muscle_group': 'chest',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['upper_chest', 'triceps'],
            'has_posture_check': False
        },
        'pike_push_ups': {
            'name': 'Pike Push-ups',
            'muscle_group': 'shoulders',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['front_delts', 'triceps'],
            'has_posture_check': False
        },
        'face_pulls': {
            'name': 'Face Pulls',
            'muscle_group': 'shoulders',
            'type': 'gym',
            'equipment': ['resistance_bands', 'cables'],
            'targets': ['rear_delts', 'rhomboids'],
            'has_posture_check': False
        },
        'reverse_flyes': {
            'name': 'Reverse Flyes',
            'muscle_group': 'shoulders',
            'type': 'gym',
            'equipment': ['dumbbells'],
            'targets': ['rear_delts'],
            'has_posture_check': False
        },
        'lunges': {
            'name': 'Lunges',
            'muscle_group': 'legs',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['quadriceps', 'glutes'],
            'has_posture_check': False
        },
        'planks': {
            'name': 'Planks',
            'muscle_group': 'core',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['core', 'shoulders'],
            'has_posture_check': False
        },
        'mountain_climbers': {
            'name': 'Mountain Climbers',
            'muscle_group': 'cardio',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['core', 'cardio'],
            'has_posture_check': False
        },
        'burpees': {
            'name': 'Burpees',
            'muscle_group': 'full_body',
            'type': 'bodyweight',
            'equipment': [],
            'targets': ['full_body', 'cardio'],
            'has_posture_check': False
        },
        'jumping_jacks': {
            'name': 'Jumping Jacks',
            'muscle_group': 'cardio',
            'type': 'cardio',
            'equipment': [],
            'targets': ['cardio', 'legs'],
            'has_posture_check': False
        }
    }

def filter_exercises_by_health_conditions(exercises, health_conditions):
    """Filter out exercises that may worsen health conditions"""
    filtered_exercises = {}
    
    # Define exercise restrictions based on health conditions
    restrictions = {
        'knee_pain': ['squats', 'lunges', 'jumping_jacks', 'burpees'],
        'shoulder_injury': ['push-ups', 'side_raises', 'pike_push_ups'],
        'back_pain': ['burpees', 'mountain_climbers'],
        'asthma': [],  # Can do all exercises but with modifications
        'heart_condition': ['burpees', 'mountain_climbers', 'jumping_jacks']
    }
    
    for exercise_key, exercise_data in exercises.items():
        should_include = True
        
        for condition in health_conditions:
            condition_lower = condition.lower().replace(' ', '_')
            if condition_lower in restrictions:
                if exercise_key in restrictions[condition_lower]:
                    should_include = False
                    break
        
        if should_include:
            filtered_exercises[exercise_key] = exercise_data
    
    return filtered_exercises

def filter_exercises_by_equipment(exercises, available_equipment):
    """Filter exercises based on available equipment"""
    if not available_equipment:
        # If no equipment specified, return bodyweight exercises only
        return {k: v for k, v in exercises.items() if v['type'] == 'bodyweight'}
    
    filtered_exercises = {}
    available_equipment_lower = [eq.lower() for eq in available_equipment]
    
    for exercise_key, exercise_data in exercises.items():
        if not exercise_data['equipment']:  # Bodyweight exercises
            filtered_exercises[exercise_key] = exercise_data
        else:
            # Check if user has required equipment
            has_equipment = any(eq in available_equipment_lower for eq in exercise_data['equipment'])
            if has_equipment:
                filtered_exercises[exercise_key] = exercise_data
    
    return filtered_exercises

def prioritize_weak_muscles(exercises, weak_muscles):
    """Prioritize exercises that target weak muscle groups"""
    if not weak_muscles:
        return exercises
    
    weak_muscles_lower = [muscle.lower().replace(' ', '_') for muscle in weak_muscles]
    prioritized = []
    regular = []
    
    for exercise_key, exercise_data in exercises.items():
        targets_weak_muscle = any(
            weak_muscle in exercise_data['targets'] or 
            weak_muscle in exercise_data['muscle_group']
            for weak_muscle in weak_muscles_lower
        )
        
        if targets_weak_muscle:
            prioritized.append((exercise_key, exercise_data))
        else:
            regular.append((exercise_key, exercise_data))
    
    # Return prioritized exercises first, then regular ones
    return dict(prioritized + regular)

def get_sets_reps_by_goal_and_level(goal, fitness_level):
    """Get sets and reps based on goal and fitness level"""
    base_config = {
        'bulking': {'sets': 4, 'reps': '8-12', 'rest': '60-90s'},
        'cutting': {'sets': 3, 'reps': '12-15', 'rest': '45-60s'},
        'maintaining': {'sets': 3, 'reps': '10-12', 'rest': '60s'},
        'strength': {'sets': 4, 'reps': '6-8', 'rest': '90-120s'},
        'endurance': {'sets': 3, 'reps': '15-20', 'rest': '30-45s'}
    }
    
    config = base_config.get(goal, base_config['maintaining'])
    
    # Adjust for fitness level
    if fitness_level == 'beginner':
        config['sets'] = max(2, config['sets'] - 1)
    elif fitness_level == 'advanced':
        config['sets'] = config['sets'] + 1
    
    return config

def recommend_workout_plan(user_profile):
    """
    Generate a personalized 7-day workout plan
    
    Args:
        user_profile (dict): User profile containing:
            - age: int
            - gender: str ('M', 'F', 'O')
            - height_cm: int
            - weight_kg: float
            - goal: str ('bulking', 'cutting', 'maintaining', 'strength', 'endurance')
            - fitness_level: str ('beginner', 'intermediate', 'advanced')
            - health_conditions: list of str
            - equipment_available: list of str
            - calories_per_day: int (optional)
            - weak_muscles: list of str (e.g., ['upper_chest', 'rear_delts'])
    
    Returns:
        dict: 7-day workout plan with daily workouts
    """
    
    # Calculate basic metrics
    bmi = calculate_bmi(user_profile['height_cm'], user_profile['weight_kg'])
    tdee = calculate_tdee(
        user_profile['age'], 
        user_profile['gender'], 
        user_profile['height_cm'], 
        user_profile['weight_kg'], 
        user_profile['fitness_level']
    )
    
    # Get exercise database
    all_exercises = get_exercise_database()
    
    # Filter exercises based on health conditions
    safe_exercises = filter_exercises_by_health_conditions(
        all_exercises, 
        user_profile.get('health_conditions', [])
    )
    
    # Filter by available equipment
    available_exercises = filter_exercises_by_equipment(
        safe_exercises, 
        user_profile.get('equipment_available', [])
    )
    
    # Prioritize weak muscles
    prioritized_exercises = prioritize_weak_muscles(
        available_exercises, 
        user_profile.get('weak_muscles', [])
    )
    
    # Get sets/reps configuration
    sets_reps_config = get_sets_reps_by_goal_and_level(
        user_profile['goal'], 
        user_profile['fitness_level']
    )
    
    # Create 7-day plan structure
    workout_plan = {
        'user_stats': {
            'bmi': bmi,
            'bmi_category': get_bmi_category(bmi),
            'tdee': tdee,
            'recommended_calories': get_recommended_calories(tdee, user_profile['goal'])
        },
        'weekly_plan': {}
    }
    
    # Define workout split based on fitness level
    if user_profile['fitness_level'] == 'beginner':
        workout_split = create_beginner_split(prioritized_exercises, sets_reps_config)
    elif user_profile['fitness_level'] == 'intermediate':
        workout_split = create_intermediate_split(prioritized_exercises, sets_reps_config)
    else:  # advanced
        workout_split = create_advanced_split(prioritized_exercises, sets_reps_config)
    
    workout_plan['weekly_plan'] = workout_split
    
    return workout_plan

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_recommended_calories(tdee, goal):
    """Get recommended daily calories based on goal"""
    if goal == 'bulking':
        return tdee + 300
    elif goal == 'cutting':
        return tdee - 500
    else:  # maintaining, strength, endurance
        return tdee

def create_beginner_split(exercises, config):
    """Create beginner-friendly 3-day split with rest days"""
    exercise_list = list(exercises.items())
    
    return {
        'Day 1': {
            'workout_name': 'Full Body A',
            'muscle_group': 'Full Body',
            'type': 'beginner',
            'exercises': [
                create_exercise_entry(exercise_list[0], config),
                create_exercise_entry(exercise_list[1], config),
                create_exercise_entry(exercise_list[2], config) if len(exercise_list) > 2 else None
            ]
        },
        'Day 2': {'workout_name': 'Rest Day', 'type': 'rest'},
        'Day 3': {
            'workout_name': 'Full Body B',
            'muscle_group': 'Full Body',
            'type': 'beginner',
            'exercises': [
                create_exercise_entry(exercise_list[3], config) if len(exercise_list) > 3 else create_exercise_entry(exercise_list[0], config),
                create_exercise_entry(exercise_list[4], config) if len(exercise_list) > 4 else create_exercise_entry(exercise_list[1], config),
                create_exercise_entry(exercise_list[0], config)
            ]
        },
        'Day 4': {'workout_name': 'Rest Day', 'type': 'rest'},
        'Day 5': {
            'workout_name': 'Full Body C',
            'muscle_group': 'Full Body',
            'type': 'beginner',
            'exercises': [
                create_exercise_entry(exercise_list[1], config),
                create_exercise_entry(exercise_list[2], config) if len(exercise_list) > 2 else create_exercise_entry(exercise_list[0], config),
                create_exercise_entry(exercise_list[5], config) if len(exercise_list) > 5 else create_exercise_entry(exercise_list[1], config)
            ]
        },
        'Day 6': {'workout_name': 'Rest Day', 'type': 'rest'},
        'Day 7': {'workout_name': 'Rest Day', 'type': 'rest'}
    }

def create_intermediate_split(exercises, config):
    """Create intermediate 4-day upper/lower split"""
    exercise_list = list(exercises.items())
    upper_exercises = [ex for ex in exercise_list if ex[1]['muscle_group'] in ['chest', 'shoulders', 'arms']]
    lower_exercises = [ex for ex in exercise_list if ex[1]['muscle_group'] in ['legs']]
    core_cardio = [ex for ex in exercise_list if ex[1]['muscle_group'] in ['core', 'cardio', 'full_body']]
    
    return {
        'Day 1': {
            'workout_name': 'Upper Body A',
            'muscle_group': 'Upper Body',
            'type': 'intermediate',
            'exercises': [create_exercise_entry(ex, config) for ex in upper_exercises[:4]]
        },
        'Day 2': {
            'workout_name': 'Lower Body A',
            'muscle_group': 'Lower Body',
            'type': 'intermediate',
            'exercises': [create_exercise_entry(ex, config) for ex in lower_exercises[:3]] + 
                        [create_exercise_entry(ex, config) for ex in core_cardio[:1]]
        },
        'Day 3': {'workout_name': 'Rest Day', 'type': 'rest'},
        'Day 4': {
            'workout_name': 'Upper Body B',
            'muscle_group': 'Upper Body',
            'type': 'intermediate',
            'exercises': [create_exercise_entry(ex, config) for ex in upper_exercises[2:6]]
        },
        'Day 5': {
            'workout_name': 'Lower Body B',
            'muscle_group': 'Lower Body',
            'type': 'intermediate',
            'exercises': [create_exercise_entry(ex, config) for ex in lower_exercises[1:4]] + 
                        [create_exercise_entry(ex, config) for ex in core_cardio[1:2]]
        },
        'Day 6': {'workout_name': 'Active Recovery', 'type': 'cardio'},
        'Day 7': {'workout_name': 'Rest Day', 'type': 'rest'}
    }

def create_advanced_split(exercises, config):
    """Create advanced 6-day push/pull/legs split"""
    exercise_list = list(exercises.items())
    push_exercises = [ex for ex in exercise_list if ex[1]['muscle_group'] in ['chest', 'shoulders'] or 'triceps' in ex[1]['targets']]
    pull_exercises = [ex for ex in exercise_list if 'biceps' in ex[1]['targets'] or 'rear_delts' in ex[1]['targets']]
    leg_exercises = [ex for ex in exercise_list if ex[1]['muscle_group'] in ['legs']]
    
    return {
        'Day 1': {
            'workout_name': 'Push A',
            'muscle_group': 'Push (Chest, Shoulders, Triceps)',
            'type': 'advanced',
            'exercises': [create_exercise_entry(ex, config) for ex in push_exercises[:5]]
        },
        'Day 2': {
            'workout_name': 'Pull A',
            'muscle_group': 'Pull (Back, Biceps)',
            'type': 'advanced',
            'exercises': [create_exercise_entry(ex, config) for ex in pull_exercises[:4]]
        },
        'Day 3': {
            'workout_name': 'Legs A',
            'muscle_group': 'Legs',
            'type': 'advanced',
            'exercises': [create_exercise_entry(ex, config) for ex in leg_exercises[:4]]
        },
        'Day 4': {
            'workout_name': 'Push B',
            'muscle_group': 'Push (Chest, Shoulders, Triceps)',
            'type': 'advanced',
            'exercises': [create_exercise_entry(ex, config) for ex in push_exercises[2:7]]
        },
        'Day 5': {
            'workout_name': 'Pull B',
            'muscle_group': 'Pull (Back, Biceps)',
            'type': 'advanced',
            'exercises': [create_exercise_entry(ex, config) for ex in pull_exercises[1:5]]
        },
        'Day 6': {
            'workout_name': 'Legs B',
            'muscle_group': 'Legs',
            'type': 'advanced',
            'exercises': [create_exercise_entry(ex, config) for ex in leg_exercises[1:5]]
        },
        'Day 7': {'workout_name': 'Rest Day', 'type': 'rest'}
    }

def create_exercise_entry(exercise_tuple, config):
    """Create a formatted exercise entry for the workout plan"""
    if not exercise_tuple:
        return None
    
    exercise_key, exercise_data = exercise_tuple
    
    return {
        'name': exercise_data['name'],
        'muscle_group': exercise_data['muscle_group'],
        'sets': config['sets'],
        'reps': config['reps'],
        'rest': config['rest'],
        'type': exercise_data['type'],
        'equipment': exercise_data['equipment'],
        'targets': exercise_data['targets'],
        'has_posture_check': exercise_data['has_posture_check'],
        'contraindications': get_exercise_contraindications(exercise_key)
    }

def get_exercise_contraindications(exercise_key):
    """Get contraindications for specific exercises"""
    contraindications = {
        'squats': ['Avoid if you have knee pain or injury'],
        'push-ups': ['Avoid if you have wrist or shoulder pain'],
        'burpees': ['Avoid if you have heart conditions or severe knee issues'],
        'mountain_climbers': ['Avoid if you have back pain or heart conditions'],
        'jumping_jacks': ['Avoid if you have knee or ankle injuries']
    }
    
    return contraindications.get(exercise_key, [])

# Example usage function
def test_recommendation_system():
    """Test the recommendation system with sample data"""
    sample_profile = {
        'age': 25,
        'gender': 'M',
        'height_cm': 175,
        'weight_kg': 70,
        'goal': 'bulking',
        'fitness_level': 'intermediate',
        'health_conditions': [],
        'equipment_available': ['dumbbells', 'yoga_mat'],
        'calories_per_day': 2500,
        'weak_muscles': ['upper_chest', 'rear_delts']
    }
    
    plan = recommend_workout_plan(sample_profile)
    return plan

if __name__ == "__main__":
    # Test the system
    test_plan = test_recommendation_system()
    print("Sample workout plan generated successfully!")
    print(f"BMI: {test_plan['user_stats']['bmi']}")
    print(f"TDEE: {test_plan['user_stats']['tdee']}")
    print(f"Day 1 workout: {test_plan['weekly_plan']['Day 1']['workout_name']}")
