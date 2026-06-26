import os
import json
import uuid
import time
import cv2
import numpy as np
import requests
import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from dotenv import load_dotenv
from django.db.models import Avg, Sum

# Models
from workouts.models import UserProfile, ChatSession, ChatMessage, MealLog, FoodItem, DailySummary, PostureAnalysis, WorkoutLog, FoodPreference
# Forms
from workouts.forms import UserProfileForm, ChatMessageForm
# Services/Chatbots
from workouts.fitness_chatbot import FitnessChatbot

# Shared global states
from .shared import MEDIAPIPE_AVAILABLE, RepCounter, REP_COUNTER_AVAILABLE, WORKOUT_STATS, WORKOUT_STATS_LOCK, GEMINI_API_KEY, GEMINI_URL, NUTRITION_DATABASE

@login_required
def pose_correction(request):
    """Pose correction center with all exercises that support pose analysis"""
    
    # Available exercises with pose correction support
    pose_exercises = {
        'squats': {
            'name': 'Squats',
            'description': 'Lower body strength exercise with real-time form checking',
            'difficulty': 'Beginner',
            'muscles': ['Quadriceps', 'Glutes', 'Hamstrings'],
            'has_posture_check': True
        },
        'pushups': {
            'name': 'Push-ups', 
            'description': 'Upper body strength exercise with form analysis',
            'difficulty': 'Beginner',
            'muscles': ['Chest', 'Shoulders', 'Triceps'],
            'has_posture_check': True
        },
        'bicep_curls': {
            'name': 'Bicep Curls',
            'description': 'Arm isolation exercise with rep counting',
            'difficulty': 'Beginner', 
            'muscles': ['Biceps', 'Forearms'],
            'has_posture_check': True
        },
        'hammer_curls': {
            'name': 'Hammer Curls',
            'description': 'Arm exercise targeting different bicep muscles',
            'difficulty': 'Beginner',
            'muscles': ['Biceps', 'Forearms'],
            'has_posture_check': True
        },
        'side_raises': {
            'name': 'Side Raises',
            'description': 'Shoulder isolation exercise with form guidance',
            'difficulty': 'Beginner',
            'muscles': ['Shoulders', 'Deltoids'],
            'has_posture_check': True
        }
    }
    
    return render(request, 'pose_correction.html', {'pose_exercises': pose_exercises})


@login_required  
def posture_analysis(request):
    """Posture analysis history page"""
    analyses = PostureAnalysis.objects.filter(user=request.user)
    return render(request, 'posture_analysis.html', {'analyses': analyses})

import os
import json
import requests
from dotenv import load_dotenv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


@login_required
def workout_stats_api(request):
    """API to fetch the active user's workout statistics from the global coordinator."""
    user_id = request.user.id
    with WORKOUT_STATS_LOCK:
        stats = WORKOUT_STATS.get(user_id, {
            'workout_name': 'None',
            'rep_count': 0,
            'left_rep_count': 0,
            'right_rep_count': 0,
            'stage': 'Ready',
            'feedback': ["No active session detected."],
            'active': False
        })
    return JsonResponse(stats)


@csrf_exempt
@login_required
def save_posture_analysis(request):
    """API to save a completed posture analysis session to the database."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            exercise_name = data.get('exercise_name', 'Unknown')
            rep_count = int(data.get('rep_count', 0))
            stage = data.get('stage', 'Ready')
            feedback = data.get('feedback', '')
            if isinstance(feedback, list):
                improvement_tips = feedback
                feedback_str = ", ".join(feedback) if feedback else "No feedback provided."
            else:
                feedback_str = str(feedback)
                improvement_tips = [feedback_str] if feedback_str else []

            # Calculate a synthetic posture score based on quality/reps (e.g., 90 - 5 * bad alignment tips, min 60)
            bad_tips_count = sum(1 for tip in improvement_tips if any(kw in tip.lower() for kw in ['bad', 'wrong', 'incorrect', 'improve', 'keep', 'maintain', 'adjust']))
            posture_score = max(60.0, min(100.0, 95.0 - (bad_tips_count * 5.0)))

            # Create PostureAnalysis record
            analysis = PostureAnalysis.objects.create(
                user=request.user,
                exercise_name=exercise_name.replace("_", " ").title(),
                posture_score=posture_score,
                feedback=feedback_str,
                improvement_tips=improvement_tips
            )

            # Normalize exercise name for muscle group mapping
            exercise_slug = exercise_name.lower().strip().replace(" ", "_")
            
            # Consistent exercise-to-muscle group mapping
            EXERCISE_MUSCLE_MAPPING = {
                'squats': 'Legs',
                'pushups': 'Chest',
                'bicep_curls': 'Biceps',
                'hammer_curls': 'Biceps',
                'side_raises': 'Shoulders'
            }
            muscle_group = EXERCISE_MUSCLE_MAPPING.get(exercise_slug, 'General')
            
            # Default weight assumptions based on exercise type (0.0 for bodyweight)
            default_weight = 0.0
            if exercise_slug in ['bicep_curls', 'hammer_curls']:
                default_weight = 10.0
            elif exercise_slug == 'side_raises':
                default_weight = 5.0

            # Auto-log completed session to WorkoutLog
            from .shared import get_exercise_id_by_name, get_exercise_by_id
            name_to_match = exercise_name.replace("_", " ").title()
            exercise_id = get_exercise_id_by_name(name_to_match)
            if exercise_id:
                details = get_exercise_by_id(exercise_id)
                if details:
                    name_to_match = details["name"]
                    muscle_group = details["muscle_group"]

            WorkoutLog.objects.create(
                user=request.user,
                exercise_name=name_to_match,
                exercise_id=exercise_id,
                sets=1,
                reps=rep_count,
                weight=default_weight,
                muscle_group=muscle_group,
                duration_minutes=10
            )

            return JsonResponse({'status': 'success', 'id': analysis.id})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'}, status=405)


import datetime
from django.db.models import Avg, Sum


def extract_landmarks_and_symmetry(image_bytes):
    import mediapipe as mp
    import numpy as np
    import cv2

    mp_pose = mp.solutions.pose
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return None
    
    lm = results.pose_landmarks.landmark
    h, w = img.shape[:2]
    
    def pt(idx):
        return (lm[idx].x * w, lm[idx].y * h)
    
    def dist(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
    
    # Key measurements
    shoulder_width = dist(pt(11), pt(12))
    hip_width = dist(pt(23), pt(24))
    left_arm = dist(pt(11), pt(13)) + dist(pt(13), pt(15))
    right_arm = dist(pt(12), pt(14)) + dist(pt(14), pt(16))
    left_leg = dist(pt(23), pt(25)) + dist(pt(25), pt(27))
    right_leg = dist(pt(24), pt(26)) + dist(pt(26), pt(28))
    torso_height = dist(pt(11), pt(23))
    
    # Symmetry scores (0-100, 100 = perfect symmetry)
    arm_symmetry = round(100 - abs(left_arm - right_arm) / 
                   max(left_arm, right_arm) * 100, 1)
    leg_symmetry = round(100 - abs(left_leg - right_leg) / 
                   max(left_leg, right_leg) * 100, 1)
    
    # Shoulder to hip ratio (ideal is 1.618 for V-taper)
    taper_ratio = round(shoulder_width / hip_width, 3) if hip_width > 0 else 0
    
    return {
        "shoulder_width_px": round(shoulder_width, 1),
        "hip_width_px": round(hip_width, 1),
        "taper_ratio": taper_ratio,
        "arm_symmetry_score": arm_symmetry,
        "leg_symmetry_score": leg_symmetry,
        "torso_height_px": round(torso_height, 1),
        "landmarks_detected": True
    }



def analyse_with_gemini_vision(image_bytes, landmark_data):
    import google.generativeai as genai
    from django.conf import settings
    import base64
    import json

    api_key = os.getenv("GEMINI_API_KEY") or getattr(settings, 'GEMINI_API_KEY', None)
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        landmark_context = ""
        if landmark_data:
            landmark_context = f"""
MediaPipe measurements from this image:
- Shoulder to hip taper ratio: {landmark_data['taper_ratio']} 
  (ideal V-taper = 1.618+)
- Arm symmetry: {landmark_data['arm_symmetry_score']}%
- Leg symmetry: {landmark_data['leg_symmetry_score']}%
Use this data to support your visual analysis.
"""
        
        prompt = f"""
You are a professional physique coach and sports scientist 
performing a body composition visual assessment.

Analyse the visible muscle development in this photo and 
return ONLY a valid JSON object. No explanation, no markdown.

{landmark_context}

Score each muscle group from 1 to 10:
1-3 = underdeveloped (priority focus needed)
4-6 = moderate development
7-10 = well developed

Return this exact JSON structure:
{{
  "muscle_scores": {{
    "chest": <1-10>,
    "shoulders": <1-10>,
    "biceps": <1-10>,
    "triceps": <1-10>,
    "back_width": <1-10>,
    "back_thickness": <1-10>,
    "core": <1-10>,
    "quads": <1-10>,
    "hamstrings": <1-10>,
    "calves": <1-10>
  }},
  "weak_groups": ["list", "of", "groups", "scoring", "under", "5"],
  "dominant_groups": ["list", "of", "groups", "scoring", "7", "or", "above"],
  "body_type": "ectomorph | mesomorph | endomorph",
  "taper_assessment": "V-taper | Balanced | Narrow shoulders | Wide hips",
  "priority_recommendation": "2-3 sentence clinical recommendation focusing on the weakest 2-3 groups",
  "suggested_split": "Push Pull Legs with extra shoulder volume",
  "confidence": "high | medium | low",
  "disclaimer": "AI visual estimation only. Not a clinical assessment."
}}
"""
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode()
        }
        response = model.generate_content([prompt, image_part])
        text = response.text.strip()
        # strip markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print("Gemini vision analysis error:", e)
        return None



def landmark_only_fallback(landmark_data):
    if not landmark_data:
        return None
    
    taper = landmark_data["taper_ratio"]
    arm_sym = landmark_data["arm_symmetry_score"]
    leg_sym = landmark_data["leg_symmetry_score"]
    
    # Derive basic scores from proportions
    shoulder_score = min(10, round(taper * 4)) if taper > 0 else 5
    symmetry_note = "good" if arm_sym > 90 else "imbalance detected"
    
    return {
        "muscle_scores": {
            "chest": 5, "shoulders": shoulder_score,
            "biceps": 5, "triceps": 5,
            "back_width": max(1, shoulder_score - 1),
            "back_thickness": 5, "core": 5,
            "quads": 5, "hamstrings": 5, "calves": 4
        },
        "weak_groups": ["calves", "hamstrings"],
        "dominant_groups": ["shoulders"] if shoulder_score >= 7 else [],
        "body_type": "mesomorph",
        "taper_assessment": "V-taper" if taper >= 1.5 else "Balanced",
        "arm_symmetry": f"{arm_sym}% ({symmetry_note})",
        "leg_symmetry": f"{leg_sym}%",
        "priority_recommendation": f"Taper ratio {taper}. Arm symmetry {arm_sym}%. Focus on lagging posterior chain and calves.",
        "suggested_split": "Push Pull Legs",
        "confidence": "low",
        "disclaimer": "Landmark-based estimation only. Upload in good lighting for full AI analysis."
    }



@login_required
def analyse_body_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    photo = request.FILES.get("photo")
    if not photo:
        return JsonResponse({"error": "No photo uploaded"}, status=400)
    
    # Validate file type
    allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if photo.content_type not in allowed:
        return JsonResponse(
            {"error": "Upload JPG, PNG or WEBP only"}, status=400
        )
    
    # Validate file size (max 10MB)
    if photo.size > 10 * 1024 * 1024:
        return JsonResponse(
            {"error": "File too large. Max 10MB."}, status=400
        )
    
    image_bytes = photo.read()
    # DO NOT save photo to disk — process in memory only
    
    # Step 1: MediaPipe landmarks
    landmark_data = extract_landmarks_and_symmetry(image_bytes)
    
    # Step 2: Gemini Vision analysis
    analysis = analyse_with_gemini_vision(image_bytes, landmark_data)
    
    # Step 3: Fallback if Gemini fails
    if not analysis:
        analysis = landmark_only_fallback(landmark_data)
    
    if not analysis:
        return JsonResponse(
            {"error": "Could not detect a body in the photo. Ensure good lighting and full upper body visibility."},
            status=422
        )
    
    # Attach landmark data to response
    if landmark_data:
        analysis["landmark_data"] = landmark_data
    
    # Auto-update weak_muscles in UserProfile
    weak = analysis.get("weak_groups", [])
    if weak:
        try:
            profile = request.user.userprofile
            profile.weak_muscles = ", ".join(weak)
            profile.save()
        except UserProfile.DoesNotExist:
            UserProfile.objects.create(
                user=request.user,
                age=25,
                height=175.0,
                weight=70.0,
                gender='M',
                fitness_level='intermediate',
                primary_goal='muscle_gain',
                available_time=60,
                weak_muscles=", ".join(weak)
            )
    
    return JsonResponse({"success": True, "analysis": analysis})



@login_required
def body_analysis_view(request):
    return render(request, "body_analysis.html")


@csrf_exempt
@login_required
def transcribe_audio(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    audio_file = request.FILES.get("audio")
    if not audio_file:
        return JsonResponse({"error": "No audio"}, status=400)
    
    try:
        from groq import Groq
        from django.conf import settings
        
        api_key = getattr(settings, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
        if not api_key:
            return JsonResponse({"error": "GROQ_API_KEY not configured"}, status=500)

        client = Groq(api_key=api_key)
        transcription = client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model="whisper-large-v3",
            prompt="Gym workout: bench press, deadlift, squat, pull ups, push ups, incline dumbbell, shoulder press, bicep curl, lateral raise, Romanian deadlift, sets, reps, kg, pounds",
            response_format="text",
            language="en"
        )
        return JsonResponse({"transcript": transcription})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



