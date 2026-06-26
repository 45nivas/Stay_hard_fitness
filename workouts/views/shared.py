import cv2
import time
import numpy as np
import threading
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe enabled - pose detection ready!")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - pose detection will be disabled")

try:
    from workouts.rep_counter import RepCounter
    REP_COUNTER_AVAILABLE = True
except ImportError as e:
    REP_COUNTER_AVAILABLE = False
    print(f"RepCounter not available: {e}")
    RepCounter = None

WORKOUT_STATS = {}
WORKOUT_STATS_LOCK = threading.Lock()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

NUTRITION_DATABASE = {
    # Proteins
    'egg white': {'calories': 52, 'protein': 11, 'carbs': 0.7, 'fat': 0.2, 'fiber': 0, 'sugar': 0.7, 'sodium': 0.166},
    'whole egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0, 'sugar': 1.1, 'sodium': 0.124},
    'chicken breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0, 'sugar': 0, 'sodium': 0.074},
    'chicken thigh': {'calories': 209, 'protein': 26, 'carbs': 0, 'fat': 11, 'fiber': 0, 'sugar': 0, 'sodium': 0.077},
    'salmon': {'calories': 208, 'protein': 20, 'carbs': 0, 'fat': 13, 'fiber': 0, 'sugar': 0, 'sodium': 0.059},
    'tuna': {'calories': 144, 'protein': 30, 'carbs': 0, 'fat': 1, 'fiber': 0, 'sugar': 0, 'sodium': 0.039},
    'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15, 'fiber': 0, 'sugar': 0, 'sodium': 0.055},
    'pork': {'calories': 242, 'protein': 27, 'carbs': 0, 'fat': 14, 'fiber': 0, 'sugar': 0, 'sodium': 0.058},
    'turkey': {'calories': 135, 'protein': 30, 'carbs': 0, 'fat': 1, 'fiber': 0, 'sugar': 0, 'sodium': 0.055},
    'greek yogurt': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fat': 0.4, 'fiber': 0, 'sugar': 3.2, 'sodium': 0.046},
    'cottage cheese': {'calories': 98, 'protein': 11, 'carbs': 3.4, 'fat': 4.3, 'fiber': 0, 'sugar': 2.7, 'sodium': 0.364},
    'whey protein': {'calories': 354, 'protein': 80, 'carbs': 8, 'fat': 1.5, 'fiber': 0, 'sugar': 8, 'sodium': 0.6},
    
    # Carbohydrates
    'white rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'sugar': 0.1, 'sodium': 0.001},
    'brown rice': {'calories': 112, 'protein': 2.6, 'carbs': 23, 'fat': 0.9, 'fiber': 1.8, 'sugar': 0.4, 'sodium': 0.005},
    'oats': {'calories': 389, 'protein': 16.9, 'carbs': 66.3, 'fat': 6.9, 'fiber': 10.6, 'sugar': 0.99, 'sodium': 0.002},
    'quinoa': {'calories': 120, 'protein': 4.4, 'carbs': 22, 'fat': 1.9, 'fiber': 2.8, 'sugar': 0.9, 'sodium': 0.005},
    'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'fiber': 1.8, 'sugar': 0.6, 'sodium': 0.001},
    'white bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2, 'fiber': 2.7, 'sugar': 5.7, 'sodium': 0.491},
    'whole wheat bread': {'calories': 247, 'protein': 13, 'carbs': 41, 'fat': 4.2, 'fiber': 6, 'sugar': 5.7, 'sodium': 0.472},
    'sweet potato': {'calories': 86, 'protein': 1.6, 'carbs': 20, 'fat': 0.1, 'fiber': 3, 'sugar': 4.2, 'sodium': 0.007},
    'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1, 'fiber': 2.2, 'sugar': 0.8, 'sodium': 0.006},
    
    # Fruits
    'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'fiber': 2.6, 'sugar': 12.2, 'sodium': 0.001},
    'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4, 'sugar': 10.4, 'sodium': 0.001},
    'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'fiber': 2.4, 'sugar': 9.4, 'sodium': 0.001},
    'strawberry': {'calories': 32, 'protein': 0.7, 'carbs': 8, 'fat': 0.3, 'fiber': 2, 'sugar': 4.9, 'sodium': 0.001},
    'blueberry': {'calories': 57, 'protein': 0.7, 'carbs': 14, 'fat': 0.3, 'fiber': 2.4, 'sugar': 10, 'sodium': 0.001},
    
    # Vegetables
    'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'fiber': 2.6, 'sugar': 1.5, 'sodium': 0.033},
    'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'fiber': 2.2, 'sugar': 0.4, 'sodium': 0.079},
    'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2, 'fiber': 2.8, 'sugar': 4.7, 'sodium': 0.069},
    'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'fiber': 1.2, 'sugar': 2.6, 'sodium': 0.005},
    
    # Fats & Nuts
    'almonds': {'calories': 579, 'protein': 21, 'carbs': 22, 'fat': 50, 'fiber': 12, 'sugar': 4.4, 'sodium': 0.001},
    'peanuts': {'calories': 567, 'protein': 26, 'carbs': 16, 'fat': 49, 'fiber': 8.5, 'sugar': 4.7, 'sodium': 0.018},
    'walnuts': {'calories': 654, 'protein': 15, 'carbs': 14, 'fat': 65, 'fiber': 6.7, 'sugar': 2.6, 'sodium': 0.002},
    'olive oil': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0.002},
    'vegetable oil': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0},
    'coconut oil': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0},
    'canola oil': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0},
    'sunflower oil': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0},
    'oil': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0},
    'avocado': {'calories': 160, 'protein': 2, 'carbs': 9, 'fat': 15, 'fiber': 7, 'sugar': 0.7, 'sodium': 0.007},
    
    # Dairy
    'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1, 'fiber': 0, 'sugar': 5, 'sodium': 0.044},
    'skim milk': {'calories': 34, 'protein': 3.4, 'carbs': 5, 'fat': 0.1, 'fiber': 0, 'sugar': 5, 'sodium': 0.044},
    'cheddar cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33, 'fiber': 0, 'sugar': 0.5, 'sodium': 0.621},
    'mozzarella': {'calories': 280, 'protein': 28, 'carbs': 2.2, 'fat': 17, 'fiber': 0, 'sugar': 1, 'sodium': 0.627},
}


def calculate_e1rm(weight, reps):
    """Epley formula — industry standard, accurate for 2-10 rep range"""
    if reps <= 1:
        return weight
    return round(weight * (1 + reps / 30), 1)


# Comprehensive Exercise Registry mapping IDs to standard names, muscle groups, and aliases
EXERCISE_REGISTRY = {
    101: {
        "name": "Dumbbell Curl",
        "muscle_group": "Biceps",
        "aliases": [
            "Dumbbell Curl", "Bicep Dumbbell Curl", "Standing Dumbbell Curl", "DB Curl",
            "Dumbbell Biceps Curl", "Alternating Dumbbell Curl", "Alternating DB Curl",
            "Two Arm Dumbbell Curl", "Both Arm Curl", "Classic Dumbbell Curl",
            "Dumbell Curl", "Bicep Dumbell Curl", "Standing Dumbell Curl", "Alternating Dumbell Curl",
            "Two Arm Dumbell Curl", "Classic Dumbell Curl", "Bicep Curl", "Bicep Curls", "Biceps curls", "Biceps Curl", "db curls"
        ]
    },
    102: {
        "name": "Hammer Curl",
        "muscle_group": "Biceps",
        "aliases": [
            "Hammer Curl", "Dumbbell Hammer Curl", "Neutral Grip Curl", "Standing Hammer Curl",
            "DB Hammer Curl", "Cross Body Hammer Curl", "Dumbell Hammer Curl", "Standing Dumbell Hammer Curl"
        ]
    },
    103: {
        "name": "Barbell Curl",
        "muscle_group": "Biceps",
        "aliases": [
            "Barbell Curl", "BB Curl", "Standing Barbell Curl", "EZ Bar Curl", "EZ Curl",
            "Barbell Curls", "BB Curls", "Standing Barbell Curls", "EZ Curls"
        ]
    },
    104: {
        "name": "Bench Press",
        "muscle_group": "Chest",
        "aliases": [
            "Bench Press", "Flat Bench Press", "Barbell Bench Press", "BB Bench Press", "Flat BB Bench",
            "Bench presses", "Flat bench presses", "Barbell bench presses"
        ]
    },
    105: {
        "name": "Incline Bench Press",
        "muscle_group": "Chest",
        "aliases": [
            "Incline Bench Press", "Incline BB Press", "Incline Barbell Press", "Incline Chest Press",
            "Incline bench presses", "Incline bb presses", "Incline barbell presses"
        ]
    },
    106: {
        "name": "Dumbbell Bench Press",
        "muscle_group": "Chest",
        "aliases": [
            "DB Bench Press", "Dumbbell Bench Press", "Flat Dumbbell Press", "Flat DB Press",
            "DB bench presses", "Dumbbell bench presses", "Dumbell bench press", "Dumbell bench presses",
            "Flat dumbell press"
        ]
    },
    107: {
        "name": "Incline Dumbbell Press",
        "muscle_group": "Chest",
        "aliases": [
            "Incline DB Press", "Incline Dumbbell Press", "Incline Chest Dumbbell Press",
            "Incline db presses", "Incline dumbbell presses", "Incline chest dumbell press", "Incline dumbell press"
        ]
    },
    108: {
        "name": "Shoulder Press",
        "muscle_group": "Shoulders",
        "aliases": [
            "Shoulder Press", "Overhead Press", "Military Press", "Standing Shoulder Press", "OHP",
            "Shoulder presses", "Overhead presses", "Military presses", "Standing shoulder presses"
        ]
    },
    109: {
        "name": "Dumbbell Shoulder Press",
        "muscle_group": "Shoulders",
        "aliases": [
            "DB Shoulder Press", "Dumbbell Shoulder Press", "Seated DB Press", "Seated Shoulder Press",
            "DB shoulder presses", "Dumbbell shoulder presses", "Dumbell shoulder press", "Seated dumbell press"
        ]
    },
    110: {
        "name": "Lateral Raise",
        "muscle_group": "Shoulders",
        "aliases": [
            "Lateral Raise", "Side Raise", "DB Lateral Raise", "Side Lateral Raise",
            "Lateral raises", "Side raises", "DB lateral raises", "Side lateral raises"
        ]
    },
    111: {
        "name": "Front Raise",
        "muscle_group": "Shoulders",
        "aliases": [
            "Front Raise", "DB Front Raise", "Front Dumbbell Raise",
            "Front raises", "DB front raises", "Front dumbell raise"
        ]
    },
    112: {
        "name": "Rear Delt Fly",
        "muscle_group": "Shoulders",
        "aliases": [
            "Rear Delt Fly", "Reverse Fly", "Bent Over Reverse Fly", "Rear Fly",
            "Rear delt flies", "Reverse flies", "Bent over reverse flies", "Rear flies"
        ]
    },
    113: {
        "name": "Lat Pulldown",
        "muscle_group": "Back",
        "aliases": [
            "Lat Pulldown", "Wide Grip Lat Pulldown", "Cable Lat Pulldown", "Pulldown",
            "Lat pulldowns", "Wide grip lat pulldowns", "Cable lat pulldowns", "Pulldowns"
        ]
    },
    114: {
        "name": "Seated Cable Row",
        "muscle_group": "Back",
        "aliases": [
            "Cable Row", "Seated Cable Row", "Machine Row", "Low Row",
            "Cable rows", "Seated cable rows", "Machine rows", "Low rows"
        ]
    },
    115: {
        "name": "Bent Over Row",
        "muscle_group": "Back",
        "aliases": [
            "Barbell Row", "Bent Over Row", "BB Row", "Bent Row",
            "Barbell rows", "Bent over rows", "BB rows", "Bent rows"
        ]
    },
    116: {
        "name": "Pull Up",
        "muscle_group": "Back",
        "aliases": [
            "Pull Up", "Pull-Up", "Wide Pull Up", "Bodyweight Pull Up",
            "Pull ups", "Pull-ups", "Wide pull ups", "Bodyweight pull ups"
        ]
    },
    117: {
        "name": "Deadlift",
        "muscle_group": "Back",
        "aliases": [
            "Deadlift", "Conventional Deadlift", "Barbell Deadlift", "BB Deadlift",
            "Deadlifts", "Conventional deadlifts", "Barbell deadlifts", "BB deadlifts"
        ]
    },
    118: {
        "name": "Romanian Deadlift",
        "muscle_group": "Legs",
        "aliases": [
            "Romanian Deadlift", "RDL", "Barbell RDL", "DB Romanian Deadlift",
            "Romanian deadlifts", "RDLs", "Barbell RDLs", "DB Romanian deadlifts"
        ]
    },
    119: {
        "name": "Squat",
        "muscle_group": "Legs",
        "aliases": [
            "Squat", "Back Squat", "Barbell Squat", "BB Squat",
            "Squats", "Back squats", "Barbell squats", "BB squats"
        ]
    },
    120: {
        "name": "Front Squat",
        "muscle_group": "Legs",
        "aliases": [
            "Front Squat", "Barbell Front Squat",
            "Front squats", "Barbell front squats"
        ]
    },
    121: {
        "name": "Leg Press",
        "muscle_group": "Legs",
        "aliases": [
            "Leg Press", "Machine Leg Press", "45 Degree Leg Press",
            "Leg presses", "Machine leg presses"
        ]
    },
    122: {
        "name": "Leg Extension",
        "muscle_group": "Legs",
        "aliases": [
            "Leg Extension", "Quad Extension", "Machine Leg Extension",
            "Leg extensions", "Quad extensions", "Machine leg extensions"
        ]
    },
    123: {
        "name": "Leg Curl",
        "muscle_group": "Legs",
        "aliases": [
            "Leg Curl", "Hamstring Curl", "Lying Leg Curl", "Seated Leg Curl",
            "Leg curls", "Hamstring curls", "Lying leg curls", "Seated leg curls"
        ]
    },
    124: {
        "name": "Calf Raise",
        "muscle_group": "Legs",
        "aliases": [
            "Standing Calf Raise", "Calf Raise", "Seated Calf Raise", "Machine Calf Raise",
            "Standing calf raises", "Calf raises", "Seated calf raises", "Machine calf raises"
        ]
    },
    125: {
        "name": "Hip Thrust",
        "muscle_group": "Legs",
        "aliases": [
            "Hip Thrust", "Barbell Hip Thrust", "Glute Bridge",
            "Hip thrusts", "Barbell hip thrusts", "Glute bridges"
        ]
    },
    126: {
        "name": "Chest Fly",
        "muscle_group": "Chest",
        "aliases": [
            "Chest Fly", "Pec Deck", "Machine Fly", "Cable Fly", "Dumbbell Fly",
            "Chest flies", "Pec decks", "Machine flies", "Cable flies", "Dumbbell flies", "Dumbell fly"
        ]
    },
    127: {
        "name": "Tricep Pushdown",
        "muscle_group": "Triceps",
        "aliases": [
            "Tricep Pushdown", "Cable Pushdown", "Rope Pushdown", "Straight Bar Pushdown",
            "Tricep pushdowns", "Cable pushdowns", "Rope pushdowns", "Straight bar pushdowns"
        ]
    },
    128: {
        "name": "Overhead Tricep Extension",
        "muscle_group": "Triceps",
        "aliases": [
            "Overhead Tricep Extension", "DB Overhead Extension", "Skull Crusher",
            "Overhead tricep extensions", "DB overhead extensions", "Skull crushers"
        ]
    },
    129: {
        "name": "Dips",
        "muscle_group": "Chest",
        "aliases": [
            "Dips", "Parallel Bar Dips", "Chest Dips", "Tricep Dips",
            "Dip", "Parallel bar dip", "Chest dip", "Tricep dip"
        ]
    }
}

# Derived EXERCISE_CATALOG to maintain backward compatibility
EXERCISE_CATALOG = {details["name"]: details["aliases"] for details in EXERCISE_REGISTRY.values()}


def get_exercise_id_by_name(name):
    normalized = normalize_exercise_name(name)
    if not normalized:
        return None
    for ex_id, details in EXERCISE_REGISTRY.items():
        if details["name"].lower() == normalized.lower():
            return ex_id
    return None


def get_exercise_by_id(ex_id):
    try:
        return EXERCISE_REGISTRY[int(ex_id)]
    except (KeyError, ValueError, TypeError):
        return None


def normalize_exercise_name(name):
    if not name:
        return ""
    import re
    # 1. Clean whitespace and stutters
    cleaned = name.strip()
    for _ in range(2):
        cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)
        
    lower_name = cleaned.lower()
    
    # 2. Check catalog direct alias matches
    for canonical_name, aliases in EXERCISE_CATALOG.items():
        if lower_name in [a.lower() for a in aliases]:
            return canonical_name
            
    # 3. Check substring matches (e.g. if the input contains "bench press" as a substring of "flat barbell bench press")
    for canonical_name, aliases in EXERCISE_CATALOG.items():
        for alias in aliases:
            if len(alias.split()) >= 2 and alias.lower() in lower_name:
                return canonical_name
                
    # 4. Fallback: Title case & singularize
    cleaned = cleaned.title()
    words = cleaned.split()
    normalized_words = []
    for w in words:
        if w.lower().endswith('s') and not w.lower().endswith('ss'):
            normalized_words.append(w[:-1])
        else:
            normalized_words.append(w)
    return " ".join(normalized_words)


def get_exercise_pr_history(user, exercise_id_or_name, limit=20):
    """
    Returns a list of dicts, one per workout session for this exercise.
    Each entry = best estimated 1RM across all sets that session.
    Also flags if that session was an all-time PR.
    """
    from workouts.models import WorkoutLog, SetLog
    import django.db.models as db_models
    
    exercise_id = None
    exercise_name = ""
    
    # Try parsing as integer ID first
    try:
        val = int(exercise_id_or_name)
        exercise_id = val
        details = get_exercise_by_id(val)
        if details:
            exercise_name = details["name"]
    except (ValueError, TypeError):
        # It's a string name
        exercise_name = str(exercise_id_or_name)
        exercise_id = get_exercise_id_by_name(exercise_name)

    q_objects = db_models.Q()
    if exercise_id:
        q_objects |= db_models.Q(exercise_id=exercise_id)
        
    # Also search by name variations for backward-compatibility / legacy data support
    normalized_name = normalize_exercise_name(exercise_name)
    variations = {exercise_name, normalized_name}
    
    # Check if this name belongs to any catalog group
    lower_normalized = normalized_name.lower()
    for canonical_name, aliases in EXERCISE_CATALOG.items():
        lower_aliases = [a.lower() for a in aliases]
        if lower_normalized == canonical_name.lower() or lower_normalized in lower_aliases or exercise_name.lower() in lower_aliases:
            variations.add(canonical_name)
            variations.add(canonical_name.title())
            for item in aliases:
                variations.add(item)
                variations.add(item.title())
                variations.add(normalize_exercise_name(item))
                
    # Add common plural suffix variations
    for v in list(variations):
        v_norm = normalize_exercise_name(v)
        if v_norm.endswith('Curl') or v_norm.endswith('Up') or v_norm.endswith('Raise') or v_norm.endswith('Squat') or v_norm.endswith('Lift') or v_norm.endswith('Dip') or v_norm.endswith('Extension'):
            variations.add(v_norm + 's')
        elif v_norm.endswith('Fly'):
            variations.add(v_norm[:-3] + 'flies')
            variations.add(v_norm + 's')
        
    # Add stuttering variations for matching old/unnormalized records
    for v in list(variations):
        words = v.split()
        if words:
            variations.add(f"{words[0]} {v}")
            
    for var in variations:
        q_objects |= db_models.Q(exercise_name__iexact=var)
        
    logs = WorkoutLog.objects.filter(
        q_objects,
        user=user
    ).order_by('-date', '-id')[:limit]

    # Reverse so it's chronological for the chart
    logs = list(reversed(logs))

    history = []
    all_time_best_e1rm = 0.0

    for log in logs:
        sets = SetLog.objects.filter(workout_log=log)
        if not sets.exists():
            # fallback to WorkoutLog avg if no SetLog data
            best_e1rm = calculate_e1rm(log.weight, log.reps)
            best_weight = log.weight
            best_reps = log.reps
        else:
            best_e1rm = 0.0
            best_weight = 0.0
            best_reps = 0
            for s in sets:
                e1rm = calculate_e1rm(s.weight, s.reps)
                if e1rm > best_e1rm:
                    best_e1rm = e1rm
                    best_weight = s.weight
                    best_reps = s.reps

        is_pr = best_e1rm > all_time_best_e1rm
        if is_pr:
            all_time_best_e1rm = best_e1rm

        # Calculate total volume for the session
        if not sets.exists():
            volume = round(log.sets * log.reps * log.weight, 1)
        else:
            volume = round(sum(s.reps * s.weight for s in sets), 1)

        history.append({
            "date": log.date.strftime("%d %b") if log.date else "",
            "date_full": log.date.strftime("%Y-%m-%d") if log.date else "",
            "e1rm": best_e1rm,
            "best_weight": best_weight,
            "best_reps": int(best_reps),
            "is_pr": is_pr,
            "volume": volume,
        })

    return history, all_time_best_e1rm

