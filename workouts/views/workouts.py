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
def workout_selection(request):
    workouts = [
        {'name': 'Squats', 'slug': 'squats'},
        {'name': 'Push-ups', 'slug': 'pushups'},
        {'name': 'Bicep Curls', 'slug': 'bicep_curls'},
        {'name': 'Hammer Curls', 'slug': 'hammer_curls'},
        {'name': 'Side Raises', 'slug': 'side_raises'},
    ]
    return render(request, 'workout_selection.html', {
        'workouts': workouts
    })


@login_required
def workout_page(request, workout_name):
    return render(request, 'workout_page.html', {'workout_name': workout_name})


def gen_frames(workout_name, user_id=None):
    # Initialize MediaPipe and rep counter
    # Try different camera indices and DirectShow backend (often required on Windows)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    # Check if camera is available
    camera_available = cap.isOpened()
    
    if user_id:
        with WORKOUT_STATS_LOCK:
            WORKOUT_STATS[user_id] = {
                'workout_name': workout_name,
                'rep_count': 0,
                'left_rep_count': 0,
                'right_rep_count': 0,
                'stage': 'Ready',
                'feedback': ["Calibrating camera..."],
                'active': True
            }
            
    if MEDIAPIPE_AVAILABLE:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        
        # Initialize rep counter for specific workout (if available)
        if REP_COUNTER_AVAILABLE:
            counter = RepCounter(workout_name)
        else:
            counter = None
    else:
        # Fallback when MediaPipe is not available
        pose = None
        mp_drawing = None
        counter = None
    
    try:
        # If no camera available, create a demo/placeholder frame
        if not camera_available:
            # Create a black placeholder frame with instructions
            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add gym-themed background color
            placeholder_frame[:] = (45, 45, 65)  # Dark blue-gray
            
            # Add title
            cv2.putText(placeholder_frame, 'STAY HARD FITNESS', 
                       (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            # Add workout name
            cv2.putText(placeholder_frame, f'{workout_name.replace("_", " ").title()} Demo', 
                       (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Add camera info
            cv2.putText(placeholder_frame, 'Camera not available in cloud environment', 
                       (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Add instructions
            cv2.putText(placeholder_frame, 'For live pose detection:', 
                       (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(placeholder_frame, '1. Run locally with webcam', 
                       (140, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(placeholder_frame, '2. Grant camera permissions', 
                       (130, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(placeholder_frame, '3. Ensure good lighting', 
                       (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Simulated training reps tracking
            demo_reps = 0
            stages = ["down", "up"]
            current_stage_idx = 0
            last_change = time.time()
            
            # Continuously yield the placeholder frame with simulated reps
            while True:
                now = time.time()
                if now - last_change > 3:
                    current_stage_idx = (current_stage_idx + 1) % 2
                    if stages[current_stage_idx] == "up":
                        demo_reps += 1
                    last_change = now
                
                frame_to_send = placeholder_frame.copy()
                cv2.putText(frame_to_send, f'Demo Mode: Reps: {demo_reps}', 
                           (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_to_send, f'Stage: {stages[current_stage_idx].upper()}', 
                           (200, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if user_id:
                    with WORKOUT_STATS_LOCK:
                        WORKOUT_STATS[user_id] = {
                            'workout_name': workout_name,
                            'rep_count': demo_reps,
                            'left_rep_count': 0,
                            'right_rep_count': 0,
                            'stage': stages[current_stage_idx].upper(),
                            'feedback': ["Simulated training active", "Maintain solid posture vector", "Eccentric contraction control"],
                            'active': True
                        }
                
                ret, buffer = cv2.imencode('.jpg', frame_to_send)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)
        
        # Normal camera processing
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            if MEDIAPIPE_AVAILABLE and pose:
                # Convert BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                
                # Convert back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Process landmarks for rep counting
                if results.pose_landmarks and counter:
                    counter.process_frame(results.pose_landmarks)
                    
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                
                # Add workout info overlay
                cv2.rectangle(image, (0, 0), (450, 140), (0, 0, 0), -1)
                cv2.putText(image, f'Workout: {workout_name.replace("_", " ").title()}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Display reps based on workout type
                if counter:
                    if workout_name in ["bicep_curls", "hammer_curls"]:
                        cv2.putText(image, f'Left: {counter.left_rep_count} Right: {counter.right_rep_count}', 
                                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f'Total Reps: {counter.rep_count}', 
                                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    else:
                        cv2.putText(image, f'Reps: {counter.rep_count}', 
                                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        
                    cv2.putText(image, f'Stage: {counter.stage if counter.stage else "Ready"}', 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Add form tips
                    tips = counter.get_feedback_text()
                    for i, tip in enumerate(tips[:3]):  # Show only 3 tips
                        cv2.putText(image, tip, (10, 150 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if user_id:
                        with WORKOUT_STATS_LOCK:
                            WORKOUT_STATS[user_id].update({
                                'rep_count': counter.rep_count,
                                'left_rep_count': getattr(counter, 'left_rep_count', 0),
                                'right_rep_count': getattr(counter, 'right_rep_count', 0),
                                'stage': counter.stage or 'Ready',
                                'feedback': tips
                            })
            else:
                # Fallback display when MediaPipe is not available
                image = frame
                cv2.rectangle(image, (0, 0), (450, 100), (0, 0, 0), -1)
                cv2.putText(image, f'Workout: {workout_name.replace("_", " ").title()}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(image, 'Pose detection unavailable in this environment', 
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    except Exception as e:
        print(f"Error in video generation: {e}")
    finally:
        if user_id:
            with WORKOUT_STATS_LOCK:
                if user_id in WORKOUT_STATS:
                    WORKOUT_STATS[user_id]['active'] = False
        if counter:
            counter.cleanup()
        cap.release()


@login_required
def video_feed(request, workout_name):
    return StreamingHttpResponse(gen_frames(workout_name, request.user.id), content_type='multipart/x-mixed-replace; boundary=frame')


@login_required
def generate_adaptive_workout(request):
    """Generates an AI adaptive workout recommendation based on weak muscle groups and goal."""
    user = request.user
    
    # 1. Fetch or create a default user profile to ensure database resilience
    try:
        user_profile = UserProfile.objects.get(user=user)
    except UserProfile.DoesNotExist:
        user_profile = UserProfile.objects.create(
            user=user,
            age=25,
            height=175.0,
            weight=70.0,
            gender='M',
            fitness_level='intermediate',
            primary_goal='muscle_gain',
            available_time=60,
            weak_muscles='Biceps,Legs'
        )

    # 2. Gather training volume history from WorkoutLog
    logs = WorkoutLog.objects.filter(user=user)
    muscle_volume = {}
    for log in logs:
        muscle_volume[log.muscle_group] = muscle_volume.get(log.muscle_group, 0.0) + log.total_volume
    volume_summary_str = ", ".join([f"{k}: {v:.1f}kg" for k, v in muscle_volume.items()]) or "No training volume logged yet."

    # 3. Try to call Gemini 1.5 Flash Cloud API
    api_key = os.getenv("GEMINI_API_KEY")
    ai_success = False
    recommendation_data = None

    if api_key:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        prompt = f"""
        You are a professional clinical fitness trainer (OS Architect).
        Design a personalized 12-week workout recommendation protocol for a user with the following profile:
        - Age: {user_profile.age}
        - Gender: {user_profile.get_gender_display()}
        - Fitness Level: {user_profile.get_fitness_level_display()}
        - Primary Goal: {user_profile.get_primary_goal_display()}
        - Weak Muscle Groups: {user_profile.weak_muscles or 'None specified'}
        - Available Time: {user_profile.available_time} minutes per session
        - Recent training volume: {volume_summary_str}

        Provide a targeted routine specifically addressing the weak muscle groups with correct biomechanics.
        Provide ONLY a valid JSON response with this exact schema:
        {{
            "difficulty_level": "Beginner | Intermediate | Advanced",
            "estimated_duration": {user_profile.available_time or 60},
            "focus_areas": ["{user_profile.weak_muscles.split(',')[0] if user_profile.weak_muscles else 'General'}"],
            "recommended_exercises": {{
                "Day 1": "list of exercises with sets and reps",
                "Day 2": "list of exercises with sets and reps",
                "Day 3": "list of exercises with sets and reps"
            }}
        }}
        Do not include any other text or markdown wrappers like ```json.
        """
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        try:
            response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=12)
            if response.status_code == 200:
                resp_json = response.json()
                ai_text = resp_json['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Clean JSON markdown formatting if present
                if ai_text.startswith('```json'):
                    ai_text = ai_text[7:-3].strip()
                elif ai_text.startswith('```'):
                    ai_text = ai_text[3:-3].strip()
                
                recommendation_data = json.loads(ai_text)
                ai_success = True
        except Exception as e:
            print(f"Gemini API error in workout recommendations: {e}")

    # 4. Fallback to local high-fidelity offline protocol builder if API fails
    if not ai_success:
        weak_list = [m.strip().title() for m in user_profile.weak_muscles.split(',')] if user_profile.weak_muscles else ['General']
        recommended = {}
        if any(m in ['Legs', 'Quadriceps', 'Glutes', 'Hamstrings'] for m in weak_list):
            recommended = {
                "Day 1 (Leg Strength Focus)": "Barbell Squats (4 sets × 8 reps), Romanian Deadlifts (3 sets × 10 reps), Calf Raises (4 sets × 15 reps). Focus on knee tracking over toes.",
                "Day 2 (Upper Body Intensity)": "Push-ups (4 sets × 12 reps), Bicep Curls (3 sets × 12 reps), Lateral Raises (3 sets × 15 reps).",
                "Day 3 (Core & Recovery)": "Planks (3 sets × 60s), Hanging Leg Raises (3 sets × 12 reps), dynamic stretching."
            }
        elif any(m in ['Biceps', 'Arms', 'Forearms'] for m in weak_list):
            recommended = {
                "Day 1 (Arm Development Focus)": "Bicep Curls (4 sets × 12 reps), Hammer Curls (3 sets × 12 reps), Tricep Pushdowns (4 sets × 10 reps). Maintain locked elbows.",
                "Day 2 (Lower Body Foundation)": "Squats (4 sets × 10 reps), Dumbbell Lunges (3 sets × 12 reps per leg), Calf Raises (3 sets × 15 reps).",
                "Day 3 (Core & Rest)": "Planks (3 sets × 60s), stretching, dynamic mobility routines."
            }
        else:
            recommended = {
                "Day 1 (Push Day)": "Push-ups (4 sets × 12 reps), Overhead Press (3 sets × 10 reps), Lateral Raises (3 sets × 12 reps).",
                "Day 2 (Pull/Legs Day)": "Dumbbell Rows (4 sets × 10 reps), Squats (4 sets × 12 reps), Bicep Curls (3 sets × 12 reps).",
                "Day 3 (Active Recovery)": "20 minutes light jogging + dynamic stretching and core stability planks."
            }
            
        recommendation_data = {
            "difficulty_level": user_profile.fitness_level.title(),
            "estimated_duration": user_profile.available_time or 60,
            "focus_areas": weak_list[:2],
            "recommended_exercises": recommended
        }

    # 5. Persist the recommendation to the database
    from workouts.models import WorkoutRecommendation
    
    # Check if there is an existing recommendation, otherwise create a new one
    rec_obj = WorkoutRecommendation.objects.create(
        user_profile=user_profile,
        recommended_exercises=recommendation_data.get('recommended_exercises', {}),
        difficulty_level=recommendation_data.get('difficulty_level', 'Intermediate'),
        estimated_duration=recommendation_data.get('estimated_duration', 60),
        focus_areas=recommendation_data.get('focus_areas', [])
    )
    
    return JsonResponse({
        'status': 'success',
        'routine': rec_obj.recommended_exercises,
        'difficulty': rec_obj.difficulty_level,
        'duration': rec_obj.estimated_duration,
        'focus': rec_obj.focus_areas
    })



def parse_workout_transcript_gemini(transcript):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
        
    prompt = f"""You are a gym workout log parser. Extract exercises from 
the transcript below. Return ONLY valid JSON, no markdown.

Critical rules:
- "X sets of Y reps" means sets=X, reps=Y. Never swap them.
- Weight always comes with kg, lbs, or pounds. Extract it.
- "60kg" or "60 kg" or "60 kilos" all mean weight=60, unit=kg
- If sets not mentioned, default to 3
- If reps not mentioned, default to 10
- If weight not mentioned, default to 0
- Bodyweight exercises (pull ups, push ups, dips) default 
  weight to 0

Return this exact JSON:
{{
  "exercises": [
    {{
      "name": "Bench Press",
      "sets": 4,
      "reps": 8,
      "weight": 60,
      "unit": "kg",
      "muscle_group": "Chest"
    }}
  ]
}}

Transcript: {transcript}
"""

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        import requests
        response = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=8)
        if response.status_code == 200:
            data = response.json()
            res_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
            if res_text.startswith("```json"):
                res_text = res_text[7:-3].strip()
            elif res_text.startswith("```"):
                res_text = res_text[3:-3].strip()
            
            import json
            parsed_data = json.loads(res_text)
            raw_exercises = parsed_data.get("exercises", [])
            mapped = []
            for item in raw_exercises:
                mapped.append({
                    "exercise_name": item.get("name", ""),
                    "sets": item.get("sets", 3),
                    "reps": item.get("reps", 10),
                    "weight_value": item.get("weight", 0.0),
                    "weight_unit": item.get("unit", "kg"),
                    "muscle_group": item.get("muscle_group", "General")
                })
            return mapped
    except Exception as e:
        print("Gemini voice log parser error:", e)
    return None


def parse_workout_transcript_local_llm(text):
    import json
    from workouts.chat.engine import get_local_ollama_response
    
    system_prompt = """
You are a structured workout parser. Extract exercises performed from the user's text.
For each exercise, extract:
- exercise_name (e.g. "Squats", "Bench Press")
- sets (integer)
- reps (integer)
- weight_value (number)
- weight_unit (string, "lbs" or "kg")
- muscle_group (must be one of: "Chest", "Legs", "Biceps", "Shoulders", "Back", "General")

Ignore chatter like water breaks, rest, warmups.
Return ONLY a valid JSON array of objects. Do not include markdown or explanations.
If no exercises are found, return an empty array [].
"""
    
    response = get_local_ollama_response(system_prompt, text, model="qwen2.5:3b")
    if not response:
        return None
        
    try:
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:-3].strip()
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:-3].strip()
        return json.loads(clean_response)
    except Exception as e:
        print("Error parsing local LLM JSON response:", e, "Raw response:", response)
        return None



def parse_with_qwen(transcript: str) -> dict | None:
    import requests, json
    
    prompt = """You are a strict gym workout log parser.
Extract per-set data from speech transcripts.
Return ONLY valid JSON. No markdown. No explanation.

CRITICAL RULES:

EXERCISE NAME:
- Extract only the actual exercise name
- Never use kg, lbs, bodyweight, sets, reps as a name
- Name always appears BEFORE the numbers

PER-SET EXTRACTION:
- Each set can have different weight and reps
- "3 sets of 30kg, 25kg, 20kg" → 3 separate sets with 
  different weights
- "set 1 30kg 8 reps, set 2 25kg 6 reps" → parse each 
  set individually
- "first set 30kg, second set 25kg" → set 1 and set 2
- If all sets same weight → repeat that weight for each set
- If only total sets mentioned with one weight → 
  use same weight for all sets

SPOTTER DETECTION:
- "with spot", "with spotter", "spotted", "with help" → 
  with_spotter=true for that set
- "without spot", "unassisted" → with_spotter=false

FAILURE DETECTION:
- "till failure", "to failure", "failed" → 
  to_failure=true, reps=0 for that set
- If reps are not mentioned or missing in a set → 
  to_failure=true, reps=0 for that set

WEIGHT RULES:
- Last number before kg/lbs is always the weight
- "each hand/arm/side" → that is the per-hand weight, 
  use it as weight directly
- "bodyweight" or "bw" → weight=0

MUSCLE GROUP MAPPING:
squats, leg press, lunges, Romanian deadlift, 
calf raises → "Legs"
bench press, incline, decline, chest fly, 
push ups, dips → "Chest"
pull ups, rows, lat pulldown, deadlift → "Back"
shoulder press, lateral raise, front raise → "Shoulders"
bicep curl, hammer curl, preacher curl → "Biceps"
tricep pushdown, skull crusher, 
tricep extension → "Triceps"
anything else → "General"

CASUAL SPEECH:
Ignore all filler: "bro", "man", "yaar", "then", 
"after that", "finished with", "started with",
"with spot", emotions, and non-exercise words.

EXAMPLES:

"bench press 3 sets, first set 30kg 8 reps, 
second set 25kg 6 reps with spot, third set 
20kg 4 reps till failure"
→ {
  "name": "Bench Press",
  "muscle_group": "Chest",
  "sets": [
    {"set_number": 1, "reps": 8, "weight": 30, 
     "unit": "kg", "with_spotter": false, 
     "to_failure": false, "notes": ""},
    {"set_number": 2, "reps": 6, "weight": 25, 
     "unit": "kg", "with_spotter": true, 
     "to_failure": false, "notes": "with spotter"},
    {"set_number": 3, "reps": 4, "weight": 20, 
     "unit": "kg", "with_spotter": false, 
     "to_failure": true, "notes": "failure set"}
  ]
}

"pull ups 4 sets, first 3 sets bodyweight 10 reps, 
last set till failure with spot"
→ {
  "name": "Pull Ups",
  "muscle_group": "Back",
  "sets": [
    {"set_number": 1, "reps": 10, "weight": 0,
     "unit": "kg", "with_spotter": false,
     "to_failure": false, "notes": ""},
    {"set_number": 2, "reps": 10, "weight": 0,
     "unit": "kg", "with_spotter": false,
     "to_failure": false, "notes": ""},
    {"set_number": 3, "reps": 10, "weight": 0,
     "unit": "kg", "with_spotter": false,
     "to_failure": false, "notes": ""},
    {"set_number": 4, "reps": 0, "weight": 0,
     "unit": "kg", "with_spotter": true,
     "to_failure": true, "notes": "failure set with spotter"}
  ]
}

"incline dumbbell 3 sets 30kg 10 reps"
→ {
  "name": "Incline Dumbbell Press",
  "muscle_group": "Chest",
  "sets": [
    {"set_number": 1, "reps": 10, "weight": 30,
     "unit": "kg", "with_spotter": false,
     "to_failure": false, "notes": ""},
    {"set_number": 2, "reps": 10, "weight": 30,
     "unit": "kg", "with_spotter": false,
     "to_failure": false, "notes": ""},
    {"set_number": 3, "reps": 10, "weight": 30,
     "unit": "kg", "with_spotter": false,
     "to_failure": false, "notes": ""}
  ]
}

Return ONLY this JSON structure:
{
  "exercises": [
    {
      "name": "Exercise Name",
      "muscle_group": "Chest",
      "sets": [
        {
          "set_number": 1,
          "reps": 8,
          "weight": 30,
          "unit": "kg",
          "with_spotter": false,
          "to_failure": false,
          "notes": ""
        }
      ]
    }
  ]
}

Transcript: """ + transcript
    
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=20
        )
        text = r.json().get("response", "").strip()
        # Strip markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        # Validate structure
        if "exercises" not in parsed:
            return None
        if not isinstance(parsed["exercises"], list):
            return None
        if len(parsed["exercises"]) == 0:
            return None
        return parsed
    except Exception:
        return None


def parse_with_gemini(transcript):
    return parse_workout_transcript_gemini(transcript)


def parse_with_regex(transcript):
    return parse_workout_transcript_fallback_regex(transcript)




def parse_workout_transcript_fallback_regex(text):
    import re
    
    # Split by common sentence/clause boundaries
    segments = re.split(r'[.,;]|\band\b|\bthen\b|\blater\b|\bnext\b|\n', text, flags=re.IGNORECASE)
    results = []
    
    # Define regex patterns with weights (requires space or preposition before weight)
    p1 = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?(\d+)\s*(?:reps?|rep)?(?:\s+(?:at|@|with|of)?\s*|\s*(?:at|@|with|of)\s*)(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?\s*(?:of\s*)?([a-zA-Z\s\-]+)',
        re.IGNORECASE
    )
    p2 = re.compile(
        r'([a-zA-Z\s\-]+)\s*(\d+)\s*(?:sets?)\s*(?:of\s*)?(\d+)\s*(?:reps?|rep)?(?:\s+(?:at|@|with|of)?\s*|\s*(?:at|@|with|of)\s*)(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?',
        re.IGNORECASE
    )
    p3 = re.compile(
        r'([a-zA-Z\s\-]+)\s*(\d+)\s*(?:x|×)\s*(\d+)(?:\s+(?:at|@|with|of)?\s*|\s*(?:at|@|with|of)\s*)(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?',
        re.IGNORECASE
    )
    p4 = re.compile(
        r'(\d+)\s*(?:x|×)\s*(\d+)\s*(?:of\s*)?([a-zA-Z\s\-]+?)(?:\s+(?:at|@|with)?\s*|\s*(?:at|@|with)\s*)(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?\b',
        re.IGNORECASE
    )
    p5 = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?(\d+)\s*(?:reps?|rep)?\s*(?:of\s*)?([a-zA-Z\s\-]+?)(?:\s+(?:at|@|with)?\s*|\s*(?:at|@|with)\s*)(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?\b',
        re.IGNORECASE
    )
    p6 = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?([a-zA-Z\s\-]+?)\s*(?:with|at|@|of)\s*(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?\s*(?:for)?\s*(\d+)\s*(?:reps?|rep)?\b',
        re.IGNORECASE
    )
    p7 = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?([a-zA-Z\s\-]+?)\s*(?:for)?\s*(\d+)\s*(?:reps?|rep)?\s*(?:at|with|@|of)\s*(\d+(?:\.\d+)?)\s*(lbs|kg|pounds|lb)?\b',
        re.IGNORECASE
    )

    # Bodyweight patterns (no weight specified)
    pbw1 = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?(\d+)\s*(?:reps?|rep)?\s*(?:of|for)?\s*([a-zA-Z\s\-]+)',
        re.IGNORECASE
    )
    pbw2 = re.compile(
        r'([a-zA-Z\s\-]+)\s*(\d+)\s*(?:sets?)\s*(?:of\s*)?(\d+)\s*(?:reps?|rep)?',
        re.IGNORECASE
    )
    pbw3 = re.compile(
        r'([a-zA-Z\s\-]+)\s*(\d+)\s*(?:x|×)\s*(\d+)',
        re.IGNORECASE
    )
    pbw4 = re.compile(
        r'(\d+)\s*(?:x|×)\s*(\d+)\s*(?:of\s*)?([a-zA-Z\s\-]+)',
        re.IGNORECASE
    )
    pbw_sets_reps_split = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?([a-zA-Z\s\-]+?)\s*(?:for)?\s*(\d+)\s*(?:reps?|rep)?\b',
        re.IGNORECASE
    )
    pbw_sets_prefix = re.compile(
        r'(\d+)\s*(?:sets?)\s*(?:of\s*)?([a-zA-Z\s\-]+)',
        re.IGNORECASE
    )
    pbw5 = re.compile(
        r'(?:\bdid\s+|\bperformed\s+)?(\d+)\s*(?:reps?\s+of\s+|reps?\s+|\s+)?([a-zA-Z\s\-]+)',
        re.IGNORECASE
    )
    pbw_sets = re.compile(
        r'([a-zA-Z\s\-]+?)\s*(\d+)\s*(?:sets?)\b',
        re.IGNORECASE
    )
    pbw_reps = re.compile(
        r'([a-zA-Z\s\-]+?)\s*(\d+)\s*(?:reps?|rep)\b',
        re.IGNORECASE
    )

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
            
        match = None
        exercise_name = ""
        sets = 1
        reps = 0
        weight_value = 0.0
        weight_unit = "kg"
        
        # 1. Try weighted patterns first
        m1 = p1.search(seg)
        if m1:
            sets = int(m1.group(1))
            reps = int(m1.group(2))
            weight_value = float(m1.group(3))
            weight_unit = m1.group(4) or "kg"
            exercise_name = m1.group(5)
            match = m1
            
        if not match:
            m2 = p2.search(seg)
            if m2:
                exercise_name = m2.group(1)
                sets = int(m2.group(2))
                reps = int(m2.group(3))
                weight_value = float(m2.group(4))
                weight_unit = m2.group(5) or "kg"
                match = m2
                
        if not match:
            m3 = p3.search(seg)
            if m3:
                exercise_name = m3.group(1)
                sets = int(m3.group(2))
                reps = int(m3.group(3))
                weight_value = float(m3.group(4))
                weight_unit = m3.group(5) or "kg"
                match = m3
                
        if not match:
            m4 = p4.search(seg)
            if m4:
                sets = int(m4.group(1))
                reps = int(m4.group(2))
                exercise_name = m4.group(3)
                weight_value = float(m4.group(4))
                weight_unit = m4.group(5) or "kg"
                match = m4

        if not match:
            m5 = p5.search(seg)
            if m5:
                sets = int(m5.group(1))
                reps = int(m5.group(2))
                exercise_name = m5.group(3)
                weight_value = float(m5.group(4))
                weight_unit = m5.group(5) or "kg"
                match = m5

        if not match:
            m6 = p6.search(seg)
            if m6:
                sets = int(m6.group(1))
                exercise_name = m6.group(2)
                weight_value = float(m6.group(3))
                weight_unit = m6.group(4) or "kg"
                reps = int(m6.group(5)) if m6.group(5) else 0
                match = m6

        if not match:
            m7 = p7.search(seg)
            if m7:
                sets = int(m7.group(1))
                exercise_name = m7.group(2)
                reps = int(m7.group(3))
                weight_value = float(m7.group(4))
                weight_unit = m7.group(5) or "kg"
                match = m7

        # 2. Try bodyweight patterns
        if not match:
            mbw1 = pbw1.search(seg)
            if mbw1:
                sets = int(mbw1.group(1))
                reps = int(mbw1.group(2))
                weight_value = 0.0
                exercise_name = mbw1.group(3)
                match = mbw1

        if not match:
            mbw2 = pbw2.search(seg)
            if mbw2:
                exercise_name = mbw2.group(1)
                sets = int(mbw2.group(2))
                reps = int(mbw2.group(3))
                weight_value = 0.0
                match = mbw2

        if not match:
            mbw3 = pbw3.search(seg)
            if mbw3:
                exercise_name = mbw3.group(1)
                sets = int(mbw3.group(2))
                reps = int(mbw3.group(3))
                weight_value = 0.0
                match = mbw3

        if not match:
            mbw4 = pbw4.search(seg)
            if mbw4:
                sets = int(mbw4.group(1))
                reps = int(mbw4.group(2))
                weight_value = 0.0
                exercise_name = mbw4.group(3)
                match = mbw4

        if not match:
            mbw_srs = pbw_sets_reps_split.search(seg)
            if mbw_srs:
                sets = int(mbw_srs.group(1))
                exercise_name = mbw_srs.group(2)
                reps = int(mbw_srs.group(3))
                weight_value = 0.0
                match = mbw_srs

        if not match:
            mbw_sp = pbw_sets_prefix.search(seg)
            if mbw_sp:
                sets = int(mbw_sp.group(1))
                exercise_name = mbw_sp.group(2)
                reps = 0  # default
                weight_value = 0.0
                match = mbw_sp

        if not match:
            mbw5 = pbw5.search(seg)
            if mbw5:
                sets = 1
                reps = int(mbw5.group(1))
                weight_value = 0.0
                exercise_name = mbw5.group(2)
                if re.search(r'[a-zA-Z]', exercise_name):
                    match = mbw5

        if not match:
            mbw_sets = pbw_sets.search(seg)
            if mbw_sets:
                exercise_name = mbw_sets.group(1)
                sets = int(mbw_sets.group(2))
                reps = 0  # default
                weight_value = 0.0
                match = mbw_sets

        if not match:
            mbw_reps = pbw_reps.search(seg)
            if mbw_reps:
                exercise_name = mbw_reps.group(1)
                sets = 3  # default
                reps = int(mbw_reps.group(2))
                weight_value = 0.0
                match = mbw_reps

        # 3. Fallback: Check for known exercise names if no numbers were matched
        if not match:
            known_exercises = [
                ("incline bench press", "Chest"),
                ("decline bench press", "Chest"),
                ("bench press", "Chest"),
                ("chest fly", "Chest"),
                ("push up", "Chest"),
                ("pushup", "Chest"),
                ("push-up", "Chest"),
                ("dips", "Chest"),
                ("dip", "Chest"),
                ("squat", "Legs"),
                ("leg press", "Legs"),
                ("lunge", "Legs"),
                ("leg extension", "Legs"),
                ("calf raise", "Legs"),
                ("bicep curl", "Biceps"),
                ("hammer curl", "Biceps"),
                ("curl", "Biceps"),
                ("skull crusher", "Biceps"),
                ("skullcrusher", "Biceps"),
                ("tricep extension", "Biceps"),
                ("tricep dip", "Biceps"),
                ("shoulder press", "Shoulders"),
                ("overhead press", "Shoulders"),
                ("military press", "Shoulders"),
                ("side raise", "Shoulders"),
                ("lateral raise", "Shoulders"),
                ("front raise", "Shoulders"),
                ("lat pulldown", "Back"),
                ("deadlift", "Back"),
                ("pull up", "Back"),
                ("pullup", "Back"),
                ("pull-up", "Back"),
                ("chin up", "Back"),
                ("chinup", "Back"),
                ("row", "Back"),
            ]
            for kw, mg in known_exercises:
                if re.search(r'\b' + re.escape(kw) + r's?\b', seg, re.IGNORECASE):
                    exercise_name = kw
                    sets = 3
                    reps = 10
                    weight_value = 0.0
                    weight_unit = "kg"
                    muscle_group = mg
                    match = True
                    break

        if match:
            ex_clean = exercise_name.strip()
            ex_clean = re.sub(
                r'^(?:today\s+|i\s+went\s+to\s+the\s+gym\s+and\s+|started\s+with\s+|did\s+some\s+|did\s+|i\s+did\s+|warmed\s+up\s+with\s+)',
                '',
                ex_clean,
                flags=re.IGNORECASE
            )
            ex_clean = re.sub(
                r'(?:\s+for\s+\d+\s+sets.*|\s+sets.*|\s+reps.*|\s+warmup|\s+warm\s+up)$',
                '',
                ex_clean,
                flags=re.IGNORECASE
            ).strip()
            
            if not ex_clean:
                continue
                
            ex_lower = ex_clean.lower()
            if any(w in ex_lower for w in ["minute", "second", "hour", "rest", "water", "break"]):
                continue
                
            muscle_group = "General"
            if any(k in ex_lower for k in ["bench", "chest", "pec", "fly", "pushup", "push up", "dips", "dip"]):
                muscle_group = "Chest"
            elif any(k in ex_lower for k in ["squat", "leg", "quad", "hamstring", "calf", "lunge", "extension", "calves"]):
                muscle_group = "Legs"
            elif any(k in ex_lower for k in ["curl", "bicep"]):
                muscle_group = "Biceps"
            elif any(k in ex_lower for k in ["shoulder", "press", "delt", "raise"]):
                muscle_group = "Shoulders"
            elif any(k in ex_lower for k in ["row", "lat", "pull", "deadlift", "back", "chinup", "chin up", "pullup", "pull up"]):
                muscle_group = "Back"
                
            results.append({
                "exercise_name": ex_clean.title(),
                "sets": sets,
                "reps": reps,
                "weight_value": weight_value,
                "weight_unit": weight_unit,
                "muscle_group": muscle_group
            })
            
    return results



@login_required
def voice_log_workout_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
        
    import json
    try:
        data = json.loads(request.body)
        text = data.get("text", "").strip()
    except Exception:
        text = request.POST.get("text", "").strip()
        
    if not text:
        return JsonResponse({"error": "No workout transcript text provided"}, status=400)
        
    # Pre-process text to normalize common speech-to-text mishearings
    import re
    misheard_mapping = {
        r'\blactose\b': 'lat pulldown',
        r'\blactoses\b': 'lat pulldowns',
        r'\blap\s*pulldown\b': 'lat pulldown',
        r'\blap\s*pulldowns\b': 'lat pulldowns',
        r'\blap\s*pull\s*down\b': 'lat pulldown',
        r'\blap\s*pull\s*downs\b': 'lat pulldowns',
        r'\blat\s*pull\s*down\b': 'lat pulldown',
        r'\blat\s*pull\s*downs\b': 'lat pulldowns',
        r'\blatpulldown\b': 'lat pulldown',
        r'\blatpulldowns\b': 'lat pulldowns',
        r'\bleg\s*press\b': 'leg press',
        r'\bdead\s*lift\b': 'deadlift',
        r'\bdead\s*lifts\b': 'deadlifts',
        r'\bdid(?:n\'t)?\s+envelopes\b': 'did incline bench press',
        r'\benvelopes\b': 'incline bench press',
        r'\benvelope\b': 'incline bench press',
        r'\bany\b': 'and',
    }
    
    cleaned_text = text
    for pattern, replacement in misheard_mapping.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
    parsed_exercises = parse_workout_transcript_local_llm(cleaned_text)
    source = "local_llm"
    
    if not parsed_exercises:
        parsed_exercises = parse_workout_transcript_fallback_regex(cleaned_text)
        source = "regex_fallback"
        
    if not parsed_exercises:
        return JsonResponse({"error": "No exercises could be parsed. Try formatting like: 'Bench press 3 sets of 12 reps at 100 lbs' or bodyweight exercises like '10 push ups'."}, status=422)
        
    logged_workouts = []
    
    for item in parsed_exercises:
        try:
            exercise_name = item.get("exercise_name", "").strip()
            sets = int(item.get("sets", 1))
            reps = int(item.get("reps", 0))
            weight_val = float(item.get("weight_value", 0.0))
            weight_unit = item.get("weight_unit", "kg").strip().lower()
            muscle_group = item.get("muscle_group", "General").strip()
            
            if not exercise_name or reps <= 0:
                continue
                
            if weight_unit in ["lbs", "lb", "pounds", "pound", "pnds"]:
                weight_kg = round(weight_val * 0.45359237, 1)
            else:
                weight_kg = round(weight_val, 1)
                
            allowed_groups = ["Legs", "Chest", "Biceps", "Shoulders", "Back", "General"]
            normalized_mg = muscle_group.title()
            if normalized_mg not in allowed_groups:
                if "leg" in normalized_mg.lower() or "quad" in normalized_mg.lower():
                    normalized_mg = "Legs"
                elif "chest" in normalized_mg.lower() or "pec" in normalized_mg.lower():
                    normalized_mg = "Chest"
                elif "bicep" in normalized_mg.lower():
                    normalized_mg = "Biceps"
                elif "shoulder" in normalized_mg.lower() or "delt" in normalized_mg.lower():
                    normalized_mg = "Shoulders"
                elif "back" in normalized_mg.lower() or "lat" in normalized_mg.lower() or "row" in normalized_mg.lower():
                    normalized_mg = "Back"
                else:
                    normalized_mg = "General"
                    
            from .shared import get_exercise_id_by_name, get_exercise_by_id, normalize_exercise_name
            normalized_name = normalize_exercise_name(exercise_name)
            exercise_id = get_exercise_id_by_name(normalized_name)
            if exercise_id:
                details = get_exercise_by_id(exercise_id)
                if details:
                    exercise_name = details["name"]
                    normalized_mg = details["muscle_group"]

            log_entry = WorkoutLog.objects.create(
                user=request.user,
                exercise_name=exercise_name,
                exercise_id=exercise_id,
                sets=sets,
                reps=reps,
                weight=weight_kg,
                muscle_group=normalized_mg,
                duration_minutes=sets * 2
            )
            
            logged_workouts.append({
                "id": log_entry.id,
                "exercise_name": log_entry.exercise_name,
                "exercise_id": log_entry.exercise_id,
                "sets": log_entry.sets,
                "reps": log_entry.reps,
                "weight": log_entry.weight,
                "muscle_group": log_entry.muscle_group,
                "volume": log_entry.total_volume
            })
        except Exception as e:
            print("Error logging voice workout entry:", e)
            continue
            
    if not logged_workouts:
        return JsonResponse({"error": "No valid exercises could be saved."}, status=422)
        
    return JsonResponse({
        "success": True,
        "source": source,
        "logged": logged_workouts
    })


# ==========================================
# DECOUPLED FRONTEND REST API ENDPOINTS
# ==========================================


@login_required
def api_workout_selection(request):
    workouts = [
        {'name': 'Squats', 'slug': 'squats', 'category': 'Legs'},
        {'name': 'Push-ups', 'slug': 'pushups', 'category': 'Chest'},
        {'name': 'Bicep Curls', 'slug': 'bicep_curls', 'category': 'Biceps'},
        {'name': 'Hammer Curls', 'slug': 'hammer_curls', 'category': 'Biceps'},
        {'name': 'Side Raises', 'slug': 'side_raises', 'category': 'Shoulders'},
    ]
    return JsonResponse({"workouts": workouts})



@csrf_exempt
@login_required
def api_get_workout_logs(request):
    """Retrieve today's logged workouts for the athlete's diary"""
    import datetime
    today = datetime.date.today()
    logs = WorkoutLog.objects.filter(user=request.user, date=today).order_by('-id')
    results = []
    
    from .shared import calculate_e1rm
    from workouts.models import SetLog
    
    for log in logs:
        # Calculate today's best e1rm
        sets = SetLog.objects.filter(workout_log=log)
        if not sets.exists():
            today_best_e1rm = calculate_e1rm(log.weight, log.reps)
        else:
            today_best_e1rm = max(calculate_e1rm(s.weight, s.reps) for s in sets) if sets.exists() else 0.0
            
        # Get previous logs of this exercise before today
        if log.exercise_id:
            prev_logs = WorkoutLog.objects.filter(
                user=request.user,
                exercise_id=log.exercise_id,
                date__lt=today
            )
        else:
            prev_logs = WorkoutLog.objects.filter(
                user=request.user,
                exercise_name__iexact=log.exercise_name,
                date__lt=today
            )
        
        if not prev_logs.exists():
            # First time doing this exercise ever
            is_new_pr = True
        else:
            prev_best = 0.0
            for pl in prev_logs:
                p_sets = SetLog.objects.filter(workout_log=pl)
                if not p_sets.exists():
                    pe1rm = calculate_e1rm(pl.weight, pl.reps)
                else:
                    pe1rm = max(calculate_e1rm(ps.weight, ps.reps) for ps in p_sets) if p_sets.exists() else 0.0
                if pe1rm > prev_best:
                    prev_best = pe1rm
            is_new_pr = today_best_e1rm > prev_best
            
        results.append({
            "id": log.id,
            "exercise_name": log.exercise_name,
            "exercise_id": log.exercise_id,
            "sets": log.sets,
            "reps": log.reps,
            "weight": log.weight,
            "muscle_group": log.muscle_group,
            "volume": log.total_volume,
            "duration": log.duration_minutes,
            "is_new_pr": is_new_pr
        })
    return JsonResponse({"success": True, "workouts": results})



@csrf_exempt
@login_required
def api_delete_workout_log(request):
    """Delete a logged workout session by ID"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    import json
    try:
        data = json.loads(request.body)
        log_id = data.get("id")
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
        
    if not log_id:
        return JsonResponse({"error": "Missing ID"}, status=400)
        
    try:
        log_entry = WorkoutLog.objects.get(id=log_id, user=request.user)
        log_entry.delete()
        return JsonResponse({"success": True})
    except WorkoutLog.DoesNotExist:
        return JsonResponse({"error": "Log entry not found"}, status=404)


@csrf_exempt
@login_required
def parse_workout_voice_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    
    from .shared import normalize_exercise_name
    import json
    try:
        data = json.loads(request.body)
        text = data.get("text", "").strip()
    except Exception:
        text = request.POST.get("text", "").strip()
        
    if not text:
        return JsonResponse({"error": "No workout transcript text provided"}, status=400)
        
    import re
    misheard_mapping = {
        r'\blactose\b': 'lat pulldown',
        r'\blactoses\b': 'lat pulldowns',
        r'\blap\s*pulldown\b': 'lat pulldown',
        r'\blap\s*pulldowns\b': 'lat pulldowns',
        r'\blap\s*pull\s*down\b': 'lat pulldown',
        r'\blap\s*pull\s*downs\b': 'lat pulldowns',
        r'\blat\s*pull\s*down\b': 'lat pulldown',
        r'\blat\s*pull\s*downs\b': 'lat pulldowns',
        r'\blatpulldown\b': 'lat pulldown',
        r'\blatpulldowns\b': 'lat pulldowns',
        r'\bleg\s*press\b': 'leg press',
        r'\bdead\s*lift\b': 'deadlift',
        r'\bdead\s*lifts\b': 'deadlifts',
        r'\bdid(?:n\'t)?\s+envelopes\b': 'did incline bench press',
        r'\benvelopes\b': 'incline bench press',
        r'\benvelope\b': 'incline bench press',
        r'\bany\b': 'and',
        r'\btensor\s+of\s+interest\s+in\s+the\s+': 'bench press ',
        r'\btensor\s+of\s+interest\b': 'bench press',
        r'\b5\s+day\s+or\s+30\s+pages\b': '5 sets of 30 reps',
        r'\byou\s+kind\s+of\s+will\s+transfer\b': 'incline bench press',
        r'\bskunk\s+crushes\b': 'skull crushers',
        r'\bskunk\s+crushers\b': 'skull crushers',
        r'\bone\b': '1',
        r'\btwo\b': '2',
        r'\bthree\b': '3',
        r'\bfour\b': '4',
        r'\bfive\b': '5',
        r'\bsix\b': '6',
        r'\bseven\b': '7',
        r'\beight\b': '8',
        r'\bnine\b': '9',
        r'\bten\b': '10',
        r'\beleven\b': '11',
        r'\btwelve\b': '12',
    }
    
    cleaned_text = text
    for pattern, replacement in misheard_mapping.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
    result = parse_with_qwen(cleaned_text)
    source = "qwen"
    
    if not result:
        result = parse_with_gemini(cleaned_text)
        source = "gemini"
        
    if not result:
        result = parse_with_regex(cleaned_text)
        source = "regex_fallback"
        
    if not result:
        return JsonResponse({"error": "Could not parse any exercises from the text. Try: 'Squats 3 sets of 12 reps @ 80 kg'."}, status=422)
        
    # Return structure matching what front-end expects for preview (per-set schema)
    from .shared import get_exercise_id_by_name, get_exercise_by_id
    standardized = []
    if source == "qwen":
        raw_list = result.get("exercises", [])
        for item in raw_list:
            sets_data = []
            for set_item in item.get("sets", []):
                r_val = set_item.get("reps")
                if r_val is None or int(r_val) <= 0:
                    s_reps = 0
                    s_fail = True
                else:
                    s_reps = int(r_val)
                    s_fail = bool(set_item.get("to_failure", False))
                sets_data.append({
                    "set_number": int(set_item.get("set_number", 1)),
                    "reps": s_reps,
                    "weight_value": float(set_item.get("weight", 0.0)),
                    "weight_unit": set_item.get("unit", "kg").strip().lower(),
                    "with_spotter": bool(set_item.get("with_spotter", False)),
                    "to_failure": s_fail,
                    "notes": set_item.get("notes", "").strip()
                })
            
            if not sets_data:
                sets_data.append({
                    "set_number": 1,
                    "reps": 10,
                    "weight_value": 0.0,
                    "weight_unit": "kg",
                    "with_spotter": False,
                    "to_failure": False,
                    "notes": ""
                })
                
            ex_name = normalize_exercise_name(item.get("name", ""))
            ex_id = get_exercise_id_by_name(ex_name)
            if ex_id:
                details = get_exercise_by_id(ex_id)
                if details:
                    ex_name = details["name"]
                    mg = details["muscle_group"]
                else:
                    mg = item.get("muscle_group", "General").strip().title()
            else:
                mg = item.get("muscle_group", "General").strip().title()

            standardized.append({
                "exercise_name": ex_name,
                "exercise_id": ex_id,
                "muscle_group": mg,
                "sets": sets_data
            })
    else:
        # Flat format from Gemini / Regex fallback -> expand to sets array
        for item in result:
            sets_count = int(item.get("sets", 1))
            reps_val = item.get("reps")
            if reps_val is None or int(reps_val) <= 0:
                reps_count = 0
                is_failure = True
            else:
                reps_count = int(reps_val)
                is_failure = False
            weight_val = float(item.get("weight_value", 0.0))
            weight_unit = item.get("weight_unit", "kg").strip().lower()
            
            sets_data = []
            for i in range(1, sets_count + 1):
                sets_data.append({
                    "set_number": i,
                    "reps": reps_count,
                    "weight_value": weight_val,
                    "weight_unit": weight_unit,
                    "with_spotter": False,
                    "to_failure": is_failure,
                    "notes": "failure set" if is_failure else ""
                })
                
            ex_name = normalize_exercise_name(item.get("exercise_name", ""))
            ex_id = get_exercise_id_by_name(ex_name)
            if ex_id:
                details = get_exercise_by_id(ex_id)
                if details:
                    ex_name = details["name"]
                    mg = details["muscle_group"]
                else:
                    mg = item.get("muscle_group", "General").strip().title()
            else:
                mg = item.get("muscle_group", "General").strip().title()

            standardized.append({
                "exercise_name": ex_name,
                "exercise_id": ex_id,
                "muscle_group": mg,
                "sets": sets_data
            })
        
    return JsonResponse({
        "success": True,
        "source": source,
        "parsed": standardized
    })


@csrf_exempt
@login_required
def confirm_workout_log_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
        
    import json
    try:
        data = json.loads(request.body)
        exercises = data.get("exercises", [])
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
        
    if not exercises:
        return JsonResponse({"error": "No exercises to save"}, status=400)
        
    from workouts.models import SetLog
    import datetime
    today = datetime.date.today()
    
    from .shared import normalize_exercise_name
    logged_workouts = []
    for item in exercises:
        try:
            raw_name = item.get("exercise_name") or item.get("name", "").strip()
            exercise_name = normalize_exercise_name(raw_name)
            muscle_group = item.get("muscle_group", "General").strip()
            sets_list = item.get("sets", [])
            
            if not exercise_name:
                continue
                
            # Fallback to flat representation if sets list is missing/empty
            if not isinstance(sets_list, list) or len(sets_list) == 0:
                sets_count = int(item.get("sets", 1))
                reps_count = int(item.get("reps", 10))
                weight_val = float(item.get("weight_value") or item.get("weight", 0.0))
                weight_unit = item.get("weight_unit") or item.get("unit") or "kg"
                sets_list = []
                for i in range(1, sets_count + 1):
                    sets_list.append({
                        "set_number": i,
                        "reps": reps_count,
                        "weight": weight_val,
                        "unit": weight_unit,
                        "with_spotter": False,
                        "to_failure": False,
                        "notes": ""
                    })
            else:
                # Normalize sets
                normalized_sets = []
                for idx, s in enumerate(sets_list):
                    normalized_sets.append({
                        "set_number": int(s.get("set_number") or (idx + 1)),
                        "reps": int(s.get("reps", 10)),
                        "weight": float(s.get("weight") or s.get("weight_value", 0.0)),
                        "unit": s.get("unit") or s.get("weight_unit") or "kg",
                        "with_spotter": bool(s.get("with_spotter", False)),
                        "to_failure": bool(s.get("to_failure", False)),
                        "notes": s.get("notes", "").strip()
                    })
                sets_list = normalized_sets
                
            # Get exercise ID
            from .shared import get_exercise_id_by_name, get_exercise_by_id
            exercise_id = item.get("exercise_id")
            if not exercise_id:
                exercise_id = get_exercise_id_by_name(exercise_name)
                
            # Override standard details if standard exercise matched
            if exercise_id:
                details = get_exercise_by_id(exercise_id)
                if details:
                    exercise_name = details["name"]
                    muscle_group = details["muscle_group"]
            
            # Normalize muscle group
            allowed_groups = ["Legs", "Chest", "Biceps", "Shoulders", "Back", "General"]
            normalized_mg = muscle_group.title()
            if normalized_mg not in allowed_groups:
                if "leg" in normalized_mg.lower() or "quad" in normalized_mg.lower():
                    normalized_mg = "Legs"
                elif "chest" in normalized_mg.lower() or "pec" in normalized_mg.lower():
                    normalized_mg = "Chest"
                elif "bicep" in normalized_mg.lower():
                    normalized_mg = "Biceps"
                elif "shoulder" in normalized_mg.lower() or "delt" in normalized_mg.lower():
                    normalized_mg = "Shoulders"
                elif "back" in normalized_mg.lower() or "lat" in normalized_mg.lower() or "row" in normalized_mg.lower():
                    normalized_mg = "Back"
                else:
                    normalized_mg = "General"
            
            # Calculate averages for analytics compatibility
            total_sets = len(sets_list)
            
            # Calculate average weight in kg
            sum_weight_kg = 0.0
            sum_reps = 0
            for s in sets_list:
                w_val = s["weight"]
                w_unit = s["unit"].strip().lower()
                if w_unit in ["lbs", "lb", "pounds", "pound", "pnds"]:
                    w_kg = w_val * 0.45359237
                else:
                    w_kg = w_val
                sum_weight_kg += w_kg
                sum_reps += s["reps"]
                
            avg_weight_kg = round(sum_weight_kg / total_sets, 1) if total_sets > 0 else 0.0
            avg_reps = round(sum_reps / total_sets) if total_sets > 0 else 0
            
            log_entry = WorkoutLog.objects.create(
                user=request.user,
                exercise_name=exercise_name,
                exercise_id=exercise_id,
                sets=total_sets,
                reps=avg_reps,
                weight=avg_weight_kg,
                muscle_group=normalized_mg,
                date=today,
                duration_minutes=total_sets * 2
            )
            
            # Create SetLog entries
            for set_data in sets_list:
                SetLog.objects.create(
                    workout_log=log_entry,
                    set_number=set_data["set_number"],
                    reps=set_data["reps"],
                    weight=set_data["weight"],
                    unit=set_data["unit"],
                    with_spotter=set_data["with_spotter"],
                    to_failure=set_data["to_failure"],
                    notes=set_data["notes"]
                )
                
            logged_workouts.append({
                "id": log_entry.id,
                "exercise_name": log_entry.exercise_name,
                "exercise_id": log_entry.exercise_id,
                "sets": log_entry.sets,
                "reps": log_entry.reps,
                "weight": log_entry.weight,
                "muscle_group": log_entry.muscle_group,
                "volume": log_entry.total_volume
            })
        except Exception as e:
            print("Error confirming workout log:", e)
            continue
            
    if not logged_workouts:
        return JsonResponse({"error": "No valid exercises could be saved"}, status=422)
        
    return JsonResponse({
        "success": True,
        "logged": logged_workouts
    })


@csrf_exempt
@login_required
def exercise_progress_api(request, exercise_name):
    """
    GET /api/exercise-progress/<exercise_name>/
    Returns PR history + current PR for a given exercise.
    """
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    from .shared import get_exercise_pr_history
    history, current_pr = get_exercise_pr_history(
        request.user, exercise_name
    )
    return JsonResponse({
        "success": True,
        "exercise": exercise_name,
        "current_pr_e1rm": current_pr,
        "history": history,
    })







