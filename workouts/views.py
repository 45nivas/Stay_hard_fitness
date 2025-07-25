from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
import cv2
import mediapipe as mp
import numpy as np
from .rep_counter import RepCounter
from .models import UserProfile, ChatSession, ChatMessage
from .forms import UserProfileForm, ChatMessageForm
from .fitness_chatbot import FitnessChatbot
import json
import uuid

def home(request):
    if request.user.is_authenticated:
        return redirect('workout_selection')
    return redirect('login')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

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

def gen_frames(workout_name):
    # Initialize MediaPipe and rep counter
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize rep counter for specific workout
    counter = RepCounter(workout_name)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process landmarks for rep counting
            if results.pose_landmarks:
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
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    except Exception as e:
        print(f"Error in video generation: {e}")
    finally:
        counter.cleanup()
        cap.release()

@login_required
def video_feed(request, workout_name):
    return StreamingHttpResponse(gen_frames(workout_name), content_type='multipart/x-mixed-replace; boundary=frame')

@login_required
def fitness_chat(request):
    """Fitness chatbot interface"""
    # Get or create chat session
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={'user': request.user}
    )
    
    # Initialize chatbot
    chatbot = FitnessChatbot()
    
    # Load user profile data from Django model
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        # Sync profile data with chatbot
        chatbot.user_data = {
            'height': user_profile.height,
            'weight': user_profile.weight,
            'age': user_profile.age,
            'gender': user_profile.get_gender_display(),
            'fitness_level': user_profile.get_fitness_level_display(),
            'goals': [user_profile.get_primary_goal_display()],
            'primary_goal': user_profile.primary_goal,  # Keep the code value too
            'injuries_or_limitations': user_profile.injuries_or_limitations,
            'available_time': user_profile.available_time,
            'weak_muscles': user_profile.weak_muscles.split(',') if user_profile.weak_muscles else [],
            'equipment_available': user_profile.equipment_available.split(',') if user_profile.equipment_available else [],
            'calories_per_day': user_profile.calories_per_day,
        }
        # Update session with profile data
        session.user_data = chatbot.user_data
        session.save()
    except UserProfile.DoesNotExist:
        # Load existing user data from session if no profile exists
        if session.user_data:
            chatbot.user_data = session.user_data
    
    if request.method == 'POST':
        form = ChatMessageForm(request.POST)
        if form.is_valid():
            user_message = form.cleaned_data['message']
            
            # Get current user profile for context
            user_profile_data = None
            try:
                user_profile = UserProfile.objects.get(user=request.user)
                user_profile_data = {
                    'height': user_profile.height,
                    'weight': user_profile.weight,
                    'age': user_profile.age,
                    'gender': user_profile.get_gender_display(),
                    'fitness_level': user_profile.get_fitness_level_display(),
                    'primary_goal': user_profile.get_primary_goal_display(),
                    'primary_goal_code': user_profile.primary_goal,
                    'injuries_or_limitations': user_profile.injuries_or_limitations,
                    'available_time': user_profile.available_time,
                    'weak_muscles': user_profile.weak_muscles.split(',') if user_profile.weak_muscles else [],
                    'equipment_available': user_profile.equipment_available.split(',') if user_profile.equipment_available else [],
                    'calories_per_day': user_profile.calories_per_day,
                    'bmi': user_profile.bmi,
                    'bmi_category': user_profile.bmi_category
                }
            except UserProfile.DoesNotExist:
                pass
            
            # Process message with chatbot (pass profile data)
            bot_response = chatbot.process_message(user_message, user_profile_data)
            
            # Save chat message
            ChatMessage.objects.create(
                session=session,
                message=user_message,
                response=bot_response
            )
            
            # Update session user data
            session.user_data = chatbot.user_data
            session.save()
            
            # Return JSON response for AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'response': bot_response,
                    'user_data': chatbot.get_user_profile_summary()
                })
    else:
        form = ChatMessageForm()
    
    # Get chat history
    messages = session.messages.all()
    
    # No welcome message - straight to business
    welcome_message = None
    
    context = {
        'form': form,
        'messages': messages,
        'welcome_message': welcome_message,
        'user_data': chatbot.get_user_profile_summary(),
        'session_id': session_id
    }
    
    return render(request, 'fitness_chat.html', context)


@login_required
def profile_setup(request):
    """Setup or update user profile"""
    try:
        profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        profile = None
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            
            # Clear existing chat session so AI trainer gets updated profile
            session_id = request.session.get('chat_session_id')
            if session_id:
                try:
                    ChatSession.objects.filter(session_id=session_id, user=request.user).delete()
                    del request.session['chat_session_id']
                except:
                    pass  # Session might not exist
            
            messages.success(request, 'Profile updated successfully! Your AI trainer has been updated with your new goals.')
            return redirect('fitness_chat')
    else:
        form = UserProfileForm(instance=profile)
    
    return render(request, 'profile_setup.html', {'form': form, 'profile': profile})


@login_required
def clear_chat_session(request):
    """Clear current chat session"""
    session_id = request.session.get('chat_session_id')
    if session_id:
        try:
            session = ChatSession.objects.get(session_id=session_id, user=request.user)
            session.delete()
            del request.session['chat_session_id']
        except ChatSession.DoesNotExist:
            pass
    
    messages.success(request, 'Chat session cleared!')
    return redirect('fitness_chat')

@login_required
def one_rep_max_calculator(request):
    """One Rep Max Calculator view"""
    return render(request, 'one_rep_max.html')


# Calorie Tracking Views - Simple but Accurate (No API Keys!)
import requests
import re
import os
from dotenv import load_dotenv
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from .models import FoodItem, MealLog, DailySummary

# Load environment variables
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Comprehensive nutrition database - ChatGPT-level accuracy (per 100g)
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
    'olive oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100, 'fiber': 0, 'sugar': 0, 'sodium': 0.002},
    'avocado': {'calories': 160, 'protein': 2, 'carbs': 9, 'fat': 15, 'fiber': 7, 'sugar': 0.7, 'sodium': 0.007},
    
    # Dairy
    'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1, 'fiber': 0, 'sugar': 5, 'sodium': 0.044},
    'skim milk': {'calories': 34, 'protein': 3.4, 'carbs': 5, 'fat': 0.1, 'fiber': 0, 'sugar': 5, 'sodium': 0.044},
    'cheddar cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33, 'fiber': 0, 'sugar': 0.5, 'sodium': 0.621},
    'mozzarella': {'calories': 280, 'protein': 28, 'carbs': 2.2, 'fat': 17, 'fiber': 0, 'sugar': 1, 'sodium': 0.627},
}

def smart_food_lookup(food_name, quantity, unit):
    """Smart food lookup with fuzzy matching like ChatGPT"""
    food_lower = food_name.lower().strip()
    
    # Direct match
    if food_lower in NUTRITION_DATABASE:
        return NUTRITION_DATABASE[food_lower], "Database"
    
    # Fuzzy matching - find partial matches
    for key, data in NUTRITION_DATABASE.items():
        # Check if any word in the food name matches
        food_words = food_lower.split()
        key_words = key.split()
        
        # If any significant word matches
        if any(word in key for word in food_words if len(word) > 3) or \
           any(word in food_lower for word in key_words if len(word) > 3):
            return data, "Database"
    
    # Special cases
    if 'egg' in food_lower:
        if 'white' in food_lower:
            return NUTRITION_DATABASE['egg white'], "Database"
        else:
            return NUTRITION_DATABASE['whole egg'], "Database"
    
    return None, "AI"

def query_ollama_for_foods(text):
    """ChatGPT-style nutrition analysis using Ollama"""
    prompt = f"""You are a nutrition expert like ChatGPT. Analyze this food input: "{text}"

Parse each food item and provide accurate nutrition data like ChatGPT would. Be very precise with quantities and nutrition values.

For each food, provide nutrition for the EXACT quantity mentioned (not per 100g).

Return ONLY a valid JSON array with this format:
[{{"food": "food name", "quantity": actual_quantity, "unit": "unit", "calories": total_calories_for_quantity, "protein": total_protein_grams, "carbs": total_carbs_grams, "fat": total_fat_grams, "fiber": total_fiber_grams, "sugar": total_sugar_grams, "sodium": total_sodium_grams}}]

Examples:
- "10 egg whites" → [{{"food": "egg whites", "quantity": 10, "unit": "pieces", "calories": 170, "protein": 36, "carbs": 2, "fat": 0.5, "fiber": 0, "sugar": 2, "sodium": 0.55}}]
- "100g oats" → [{{"food": "oats", "quantity": 100, "unit": "g", "calories": 389, "protein": 16.9, "carbs": 66.3, "fat": 6.9, "fiber": 10.6, "sugar": 1, "sodium": 0.002}}]

Be as accurate as ChatGPT with nutrition data!"""

    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }
    
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        
        resp_json = resp.json()
        response_text = resp_json.get("response", "")
        
        # Extract JSON array
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            foods = json.loads(match.group())
            
            # Validate and enhance with database lookup
            enhanced_foods = []
            for food in foods:
                if isinstance(food, dict) and food.get('food'):
                    # Try database lookup first
                    db_data, source = smart_food_lookup(
                        food.get('food', ''), 
                        food.get('quantity', 100), 
                        food.get('unit', 'g')
                    )
                    
                    if db_data and source == "Database":
                        # Scale database data to actual quantity
                        quantity = float(food.get('quantity', 100))
                        
                        # Determine scaling factor
                        if 'piece' in food.get('unit', '').lower() or food.get('unit', '') == '':
                            # For pieces (like eggs), assume ~50g each
                            if 'egg' in food.get('food', '').lower():
                                scale = (quantity * 33) / 100  # 1 egg white ≈ 33g
                            else:
                                scale = (quantity * 50) / 100  # Default piece weight
                        else:
                            scale = quantity / 100  # For weight-based units
                        
                        enhanced_food = {
                            'food': food.get('food'),
                            'quantity': quantity,
                            'unit': food.get('unit', 'g'),
                            'calories': round(db_data['calories'] * scale, 1),
                            'protein': round(db_data['protein'] * scale, 1),
                            'carbs': round(db_data['carbs'] * scale, 1),
                            'fat': round(db_data['fat'] * scale, 1),
                            'fiber': round(db_data['fiber'] * scale, 1),
                            'sugar': round(db_data['sugar'] * scale, 1),
                            'sodium': round(db_data['sodium'] * scale, 3),
                            'source': 'Database'
                        }
                    else:
                        # Use AI estimate
                        enhanced_food = {
                            'food': str(food.get('food', '')),
                            'quantity': float(food.get('quantity', 100)),
                            'unit': str(food.get('unit', 'g')),
                            'calories': float(food.get('calories', 0)),
                            'protein': float(food.get('protein', 0)),
                            'carbs': float(food.get('carbs', 0)),
                            'fat': float(food.get('fat', 0)),
                            'fiber': float(food.get('fiber', 0)),
                            'sugar': float(food.get('sugar', 0)),
                            'sodium': float(food.get('sodium', 0)),
                            'source': 'AI'
                        }
                    
                    enhanced_foods.append(enhanced_food)
            
            return enhanced_foods
        else:
            return []
            
    except Exception as e:
        print("Ollama error:", e)
        return []


@login_required
def calorie_tracker(request):
    """Main calorie tracker page"""
    return render(request, 'calorie_tracker.html')


@csrf_exempt
@login_required
def log_meal_from_voice(request):
    """Simple but accurate nutrition tracking - like ChatGPT (No API keys needed!)"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        data = json.loads(request.body)
        transcribed_text = data.get("text", "")
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    if not transcribed_text:
        return JsonResponse({"error": "No transcribed text provided"}, status=400)

    # Parse and get nutrition like ChatGPT
    try:
        foods = query_ollama_for_foods(transcribed_text)
    except Exception as e:
        return JsonResponse({"error": f"Error analyzing food: {str(e)}"}, status=500)
    
    if not isinstance(foods, list) or not foods:
        return JsonResponse({"error": "Could not understand the food. Try: '2 eggs and 100g oats'"}, status=400)

    # Save each food to database
    results = []
    for food_obj in foods:
        food = food_obj.get("food")
        quantity = float(food_obj.get("quantity", 1))
        unit = food_obj.get("unit", "")
        calories = float(food_obj.get("calories", 0))
        protein = float(food_obj.get("protein", 0))
        carbs = float(food_obj.get("carbs", 0))
        fat = float(food_obj.get("fat", 0))
        fiber = float(food_obj.get("fiber", 0))
        sugar = float(food_obj.get("sugar", 0))
        sodium = float(food_obj.get("sodium", 0))
        source = food_obj.get("source", "AI")
        
        if not food:
            continue

        # Save or update FoodItem
        food_item, _ = FoodItem.objects.get_or_create(
            name=food,
            defaults={
                "calories_per_100g": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat,
                "fiber": fiber,
                "sugar": sugar,
                "sodium": sodium,
            }
        )
        
        # Update with latest data
        food_item.calories_per_100g = calories
        food_item.protein = protein
        food_item.carbs = carbs
        food_item.fat = fat
        food_item.fiber = fiber
        food_item.sugar = sugar
        food_item.sodium = sodium
        food_item.save()

        # Save MealLog
        MealLog.objects.create(
            user=request.user,
            food_item=food_item,
            quantity=quantity,
            unit=unit,
            calories=calories,
            protein=protein,
            carbs=carbs,
            fat=fat,
            fiber=fiber,
            sugar=sugar,
            sodium=sodium,
            source=source,
            date=timezone.now().date()
        )
        
        results.append(food_obj)

    # Get today's totals
    today = timezone.now().date()
    logs = MealLog.objects.filter(user=request.user, date=today)
    total_calories = sum(log.calories for log in logs)
    total_protein = sum(log.protein for log in logs)
    total_carbs = sum(log.carbs for log in logs)
    total_fats = sum(log.fat for log in logs)
    total_fiber = sum(log.fiber for log in logs)
    total_sugar = sum(log.sugar for log in logs)
    total_sodium = sum(log.sodium for log in logs)

    return JsonResponse({
        "logged": results,
        "date": str(today),
        "total_calories": total_calories,
        "total_protein": total_protein,
        "total_carbs": total_carbs,
        "total_fats": total_fats,
        "total_fiber": total_fiber,
        "total_sugar": total_sugar,
        "total_sodium": total_sodium,
    })


@csrf_exempt
@login_required
def get_daily_summary(request):
    """Get enhanced daily nutrition summary for current user"""
    if request.method == "GET":
        today = timezone.now().date()
        logs = MealLog.objects.filter(user=request.user, date=today)
        total_calories = sum(log.calories for log in logs)
        total_protein = sum(log.protein for log in logs)
        total_carbs = sum(log.carbs for log in logs)
        total_fats = sum(log.fat for log in logs)
        total_fiber = sum(log.fiber for log in logs)
        total_sugar = sum(log.sugar for log in logs)
        total_sodium = sum(log.sodium for log in logs)
        
        return JsonResponse({
            "date": str(today),
            "total_calories": total_calories,
            "total_protein": total_protein,
            "total_carbs": total_carbs,
            "total_fats": total_fats,
            "total_fiber": total_fiber,
            "total_sugar": total_sugar,
            "total_sodium": total_sodium,
        })
    return JsonResponse({"error": "GET required"}, status=400)


@csrf_exempt
@login_required
def get_daily_meals(request):
    """Get daily meals for current user with enhanced nutrition"""
    if request.method == "GET":
        today = timezone.now().date()
        logs = MealLog.objects.filter(user=request.user, date=today)
        meals = []
        for log in logs:
            meals.append({
                "id": log.id,
                "food": log.food_item.name,
                "quantity": log.quantity,
                "unit": log.unit,
                "calories": log.calories,
                "protein": log.protein,
                "carbs": log.carbs,
                "fat": log.fat,
                "fiber": log.fiber,
                "sugar": log.sugar,
                "sodium": log.sodium,
                "source": log.source,
            })
        return JsonResponse({"meals": meals})
    return JsonResponse({"error": "GET required"}, status=400)


@csrf_exempt
@login_required
def delete_meal(request):
    """Delete a meal log entry"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            meal_id = data.get("id")
            
            # Ensure user can only delete their own meals
            meal_log = MealLog.objects.get(id=meal_id, user=request.user)
            meal_log.delete()
            
            return JsonResponse({"success": True})
        except MealLog.DoesNotExist:
            return JsonResponse({"success": False, "error": "Meal not found"})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    
    return JsonResponse({"error": "POST required"}, status=400)

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
    # You can add logic here to fetch actual analysis data from database
    # For now, returning empty context
    analyses = []  # This would contain actual PostureAnalysis objects
    return render(request, 'posture_analysis.html', {'analyses': analyses})
