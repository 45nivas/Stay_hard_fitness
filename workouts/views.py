from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
import cv2
import time
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available - pose detection will be disabled")
import numpy as np
try:
    from .rep_counter import RepCounter
    REP_COUNTER_AVAILABLE = True
except ImportError as e:
    REP_COUNTER_AVAILABLE = False
    print(f"RepCounter not available: {e}")
    RepCounter = None
from .models import UserProfile, ChatSession, ChatMessage
from .forms import UserProfileForm, ChatMessageForm
from .fitness_chatbot import FitnessChatbot
import json
import uuid
import os
import requests
from dotenv import load_dotenv
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

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
    
    # Check if camera is available
    camera_available = cap.isOpened()
    
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
            
            # Add demo rep counter
            cv2.putText(placeholder_frame, 'Demo Mode: Reps: 0', 
                       (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add fitness emoji/icon
            cv2.putText(placeholder_frame, '💪 🏃‍♂️ 🏋️‍♀️', 
                       (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 100), 2)
            
            # Continuously yield the same placeholder frame
            while True:
                ret, buffer = cv2.imencode('.jpg', placeholder_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)  # Small delay to prevent overwhelming the connection
        
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
        if counter:
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
    
    # Professional chat approach - show limited recent messages or start fresh
    show_history = request.GET.get('show_history', 'false') == 'true'
    if show_history:
        # Show recent messages when explicitly requested
        messages = session.messages.all().order_by('-timestamp')[:5]
    else:
        # Start with a clean professional interface - no old history by default
        messages = []
    
    # Professional welcome message for new sessions
    if not messages and not show_history:
        welcome_message = "Hello! I'm your AI fitness trainer. I can help you with workout plans, nutrition advice, and fitness goals. How can I assist you today?"
    else:
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


@login_required
def carb_cycling_calculator(request):
    """Carb cycling calculator with BMR-based calculations"""
    result = None
    
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.POST.get('age', 0))
            gender = request.POST.get('gender', 'male')
            height = float(request.POST.get('height', 0))
            weight = float(request.POST.get('weight', 0))
            height_unit = request.POST.get('height_unit', 'cm')
            weight_unit = request.POST.get('weight_unit', 'kg')
            activity_level = request.POST.get('activity_level', 'moderate')
            goal = request.POST.get('goal', 'maintenance')
            training_days = int(request.POST.get('training_days', 3))
            
            # Convert units if needed
            if height_unit == 'ft':
                height = height * 30.48  # convert feet to cm
            if weight_unit == 'lbs':
                weight = weight * 0.453592  # convert lbs to kg
            
            # Calculate BMR using Mifflin-St Jeor formula
            if gender == 'male':
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            
            # Activity multipliers
            activity_multipliers = {
                'sedentary': 1.2,
                'light': 1.375,
                'moderate': 1.55,
                'active': 1.725,
                'very_active': 1.9
            }
            
            # Calculate TDEE
            tdee = bmr * activity_multipliers.get(activity_level, 1.55)
            
            # Adjust calories based on goal
            if goal == 'fat_loss':
                daily_calories = tdee * 0.85  # 15% deficit
            elif goal == 'muscle_gain':
                daily_calories = tdee * 1.1   # 10% surplus
            else:  # maintenance
                daily_calories = tdee
            
            # Calculate macros
            protein_g = weight * 2.2  # 2.2g per kg
            protein_calories = protein_g * 4
            
            fat_calories = daily_calories * 0.25  # 25% of calories
            fat_g = fat_calories / 9
            
            remaining_calories = daily_calories - protein_calories - fat_calories
            
            # Carb cycling calculations (convert kg to lbs for carb calculations)
            weight_lbs = weight * 2.20462
            
            # Calculate carb amounts for different days
            high_carb_g = weight_lbs * 2.25  # 2.0-2.5g per lb
            medium_carb_g = weight_lbs * 1.25  # 1.0-1.5g per lb
            low_carb_g = weight_lbs * 0.5   # 0.5g per lb
            
            # Calculate calories for each day type
            high_carb_calories = (protein_g * 4) + (fat_g * 9) + (high_carb_g * 4)
            medium_carb_calories = (protein_g * 4) + (fat_g * 9) + (medium_carb_g * 4)
            low_carb_calories = (protein_g * 4) + (fat_g * 9) + (low_carb_g * 4)
            
            # Create weekly schedule
            weekly_schedule = []
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Assign carb days based on training days
            for i, day in enumerate(days):
                if i < training_days:
                    if i < training_days // 2:
                        carb_type = 'High'
                    else:
                        carb_type = 'Medium'
                else:
                    carb_type = 'Low'
                
                weekly_schedule.append({
                    'day': day,
                    'carb_type': carb_type,
                    'carbs': high_carb_g if carb_type == 'High' else medium_carb_g if carb_type == 'Medium' else low_carb_g,
                    'calories': high_carb_calories if carb_type == 'High' else medium_carb_calories if carb_type == 'Medium' else low_carb_calories
                })
            
            result = {
                'bmr': round(bmr),
                'tdee': round(tdee),
                'daily_calories': round(daily_calories),
                'protein_g': round(protein_g, 1),
                'fat_g': round(fat_g, 1),
                'high_carb_g': round(high_carb_g, 1),
                'medium_carb_g': round(medium_carb_g, 1),
                'low_carb_g': round(low_carb_g, 1),
                'high_carb_calories': round(high_carb_calories),
                'medium_carb_calories': round(medium_carb_calories),
                'low_carb_calories': round(low_carb_calories),
                'weekly_schedule': weekly_schedule,
                'weight_display': f"{weight:.1f} {weight_unit}",
                'height_display': f"{height:.1f} {height_unit}"
            }
            
        except (ValueError, TypeError) as e:
            result = {'error': 'Please enter valid numbers for all fields.'}
    
    return render(request, 'carb_cycling.html', {'result': result})


# Calorie Tracking Views - Simple but Accurate (No API Keys!)
import re
from .models import FoodItem, MealLog, DailySummary

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

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

def query_gemini_for_foods(text):
    """ChatGPT-style nutrition analysis using Gemini API"""
    
    if not GEMINI_API_KEY:
        # Fallback to manual parsing if no API key
        return parse_food_manually(text)
    
    prompt = f"""You are a nutrition expert like ChatGPT. Analyze this food input: "{text}"

Parse each food item and provide accurate nutrition data like ChatGPT would. Be very precise with quantities and nutrition values.

For each food, provide nutrition for the EXACT quantity mentioned (not per 100g).

Return ONLY a valid JSON array with this format:
[{{"food": "food name", "quantity": actual_quantity, "unit": "unit", "calories": total_calories_for_quantity, "protein": total_protein_grams, "carbs": total_carbs_grams, "fat": total_fat_grams, "fiber": total_fiber_grams, "sugar": total_sugar_grams, "sodium": total_sodium_grams}}]

Examples:
- "10 egg whites" → [{{"food": "egg whites", "quantity": 10, "unit": "pieces", "calories": 170, "protein": 36, "carbs": 2, "fat": 0.5, "fiber": 0, "sugar": 2, "sodium": 0.55}}]
- "100g oats" → [{{"food": "oats", "quantity": 100, "unit": "g", "calories": 389, "protein": 16.9, "carbs": 66.3, "fat": 6.9, "fiber": 10.6, "sugar": 0.99, "sodium": 0.002}}]

Be as accurate as ChatGPT with nutrition data!"""

    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON array
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                foods = json.loads(match.group())
                
                # Validate and enhance with database lookup - DATABASE ALWAYS WINS
                enhanced_foods = []
                for food in foods:
                    if isinstance(food, dict) and food.get('food'):
                        food_name = food.get('food', '').lower().strip()
                        quantity = float(food.get('quantity', 100))
                        unit = food.get('unit', 'g')
                        
                        # PRIORITY 1: Check our database first (most accurate for common foods)
                        db_data, source = smart_food_lookup(food_name, quantity, unit)
                        
                        print(f"DEBUG: Food={food_name}, Quantity={quantity}, Unit={unit}")
                        print(f"DEBUG: Database lookup result - Source: {source}")
                        
                        if db_data and source == "Database":
                            # Use our corrected database values - ALWAYS prioritize this!
                            multiplier = get_quantity_multiplier(quantity, unit)
                            print(f"DEBUG: Using DATABASE - Multiplier={multiplier}, Final calories={round(db_data['calories'] * multiplier, 1)}")
                            
                            enhanced_food = {
                                'food': food.get('food'),
                                'quantity': quantity,
                                'unit': unit,
                                'calories': round(db_data['calories'] * multiplier, 1),
                                'protein': round(db_data['protein'] * multiplier, 1),
                                'carbs': round(db_data['carbs'] * multiplier, 1),
                                'fat': round(db_data['fat'] * multiplier, 1),
                                'fiber': round(db_data['fiber'] * multiplier, 1),
                                'sugar': round(db_data['sugar'] * multiplier, 1),
                                'sodium': round(db_data['sodium'] * multiplier, 3),
                                'source': 'Database'
                            }
                            print(f"DEBUG: Database result: {enhanced_food['food']} -> {enhanced_food['calories']} cal")
                        else:
                            # Use AI estimate only if database lookup fails
                            print(f"DEBUG: Using AI estimate for {food.get('food')}")
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
                                'source': 'Gemini AI'
                            }
                        
                        enhanced_foods.append(enhanced_food)
                
                return enhanced_foods
            else:
                return parse_food_manually(text)
        else:
            return parse_food_manually(text)
            
    except Exception as e:
        print("Gemini API error:", e)
        return parse_food_manually(text)

def parse_food_manually(text):
    """Manual food parsing fallback when APIs are unavailable"""
    # Simple regex parsing for common patterns
    import re
    
    # Pattern: number + unit + food
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s+(.+?)(?:\s+and\s+|$|,)'
    matches = re.findall(pattern, text)
    
    foods = []
    if matches:
        for match in matches:
            quantity, unit, food_name = match
            
            # Try database lookup
            db_data, source = smart_food_lookup(food_name.strip(), float(quantity), unit)
            
            if db_data:
                food_item = {
                    'food': food_name.strip(),
                    'quantity': float(quantity),
                    'unit': unit if unit else 'g',
                    'calories': db_data.get('calories', 100),
                    'protein': db_data.get('protein', 5),
                    'carbs': db_data.get('carbs', 15),
                    'fat': db_data.get('fat', 3),
                    'fiber': db_data.get('fiber', 2),
                    'sugar': db_data.get('sugar', 1),
                    'sodium': db_data.get('sodium', 0.1),
                    'source': 'Manual Parse'
                }
                foods.append(food_item)
    
    # If no matches, try simple fallback
    if not foods:
        foods = [{
            'food': text,
            'quantity': 1,
            'unit': 'serving',
            'calories': 150,
            'protein': 5,
            'carbs': 20,
            'fat': 3,
            'fiber': 2,
            'sugar': 1,
            'sodium': 0.2,
            'source': 'Estimated'
        }]
    
    return foods


@login_required
def calorie_tracker(request):
    """Main calorie tracker page"""
    return render(request, 'calorie_tracker.html')


@csrf_exempt
@login_required  
def recalculate_meals(request):
    """Recalculate all meals with updated nutrition database - fixes oil calories"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        # Get all meals for today
        from django.utils import timezone
        today = timezone.now().date()
        meals = Meal.objects.filter(user=request.user, created_at__date=today)
        
        updated_count = 0
        for meal in meals:
            # Recalculate nutrition using current database
            db_data, source = smart_food_lookup(meal.food_name, meal.quantity, meal.unit)
            
            if db_data and source == "Database":
                # Recalculate with correct values
                multiplier = get_quantity_multiplier(meal.quantity, meal.unit)
                
                meal.calories = round(db_data['calories'] * multiplier, 1)
                meal.protein = round(db_data['protein'] * multiplier, 1)
                meal.carbs = round(db_data['carbs'] * multiplier, 1)
                meal.fat = round(db_data['fat'] * multiplier, 1)
                meal.fiber = round(db_data['fiber'] * multiplier, 1)
                meal.sugar = round(db_data['sugar'] * multiplier, 1)
                meal.sodium = round(db_data['sodium'] * multiplier, 3)
                meal.source = 'Database (Updated)'
                
                meal.save()
                updated_count += 1
                print(f"DEBUG: Updated {meal.food_name} - {meal.quantity}{meal.unit} -> {meal.calories} calories")
        
        return JsonResponse({
            "success": True,
            "message": f"Updated {updated_count} meals with corrected nutrition data",
            "updated_count": updated_count
        })
        
    except Exception as e:
        return JsonResponse({"error": f"Error recalculating meals: {str(e)}"}, status=500)


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
        foods = query_gemini_for_foods(transcribed_text)
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
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def smart_food_lookup_with_gemini(food_name, quantity, unit):
    """Enhanced food lookup with Gemini API fallback for gym nutrition tracking"""
    food_lower = food_name.lower().strip()
    
    # Step 1: Try exact database match (fastest for common gym foods)
    if food_lower in NUTRITION_DATABASE:
        nutrition = NUTRITION_DATABASE[food_lower].copy()
        multiplier = get_quantity_multiplier(quantity, unit)
        for key in ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']:
            if key in nutrition:
                nutrition[key] = round(nutrition[key] * multiplier, 1)
        return nutrition, "Database"
    
    # Step 2: Try fuzzy matching for similar foods
    for db_food in NUTRITION_DATABASE.keys():
        if food_lower in db_food or db_food in food_lower:
            nutrition = NUTRITION_DATABASE[db_food].copy()
            multiplier = get_quantity_multiplier(quantity, unit)
            for key in ['calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar']:
                if key in nutrition:
                    nutrition[key] = round(nutrition[key] * multiplier, 1)
            return nutrition, "Fuzzy Match"
    
    # Step 3: Use Gemini AI for unknown foods
    return query_gemini_nutrition(food_name, quantity, unit), "Gemini AI"

def query_gemini_nutrition(food_name, quantity, unit):
    """Query Gemini API for nutrition information - perfect for fitness tracking"""
    
    if not GEMINI_API_KEY:
        return get_generic_fallback(quantity, unit)
    
    prompt = f"""
    You are a fitness nutrition expert. Calculate accurate nutrition information for: {quantity} {unit} of {food_name}

    Provide ONLY a valid JSON response with these exact keys for gym nutrition tracking:
    {{
        "calories": <number>,
        "protein": <number in grams>,
        "carbs": <number in grams>,
        "fat": <number in grams>,
        "fiber": <number in grams>,
        "sugar": <number in grams>
    }}

    Use USDA nutrition database standards. Focus on accuracy for fitness goals.
    """
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            ai_text = data['candidates'][0]['content']['parts'][0]['text']
            
            # Clean JSON response
            ai_text = ai_text.strip()
            if ai_text.startswith('```json'):
                ai_text = ai_text[7:-3]
            elif ai_text.startswith('```'):
                ai_text = ai_text[3:-3]
            
            nutrition_data = json.loads(ai_text)
            return nutrition_data
                
        else:
            return get_generic_fallback(quantity, unit)
            
    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_generic_fallback(quantity, unit)

def get_quantity_multiplier(quantity, unit):
    """Convert quantity to multiplier for 100g base nutrition values"""
    try:
        qty = float(quantity)
        
        # Convert common gym food units to grams
        unit_lower = unit.lower()
        if unit_lower in ['g', 'grams', 'gram']:
            return qty / 100
        elif unit_lower in ['kg', 'kilograms', 'kilogram']:
            return qty * 10  # 1kg = 1000g, so 1000/100 = 10
        elif unit_lower in ['cup', 'cups']:
            return qty * 2.4  # Average 240g per cup
        elif unit_lower in ['slice', 'slices']:
            return qty * 0.3  # Average 30g per slice
        elif unit_lower in ['piece', 'pieces', 'pcs', 'whole']:
            # Handle specific foods
            if 'egg' in str(quantity).lower():
                return qty * 0.5  # Average egg ~50g
            else:
                return qty * 1.0  # Default 100g per piece
        elif unit_lower in ['tbsp', 'tablespoon', 'tablespoons']:
            return qty * 0.15  # ~15g per tablespoon
        elif unit_lower in ['tsp', 'teaspoon', 'teaspoons']:
            return qty * 0.05  # ~5g per teaspoon
        else:
            return qty / 100  # Default: treat as grams
            
    except (ValueError, TypeError):
        return 1.0  # Default multiplier

def get_generic_fallback(quantity=1, unit="serving"):
    """Generic fallback nutrition when Gemini API fails"""
    base_nutrition = {
        "calories": 150,
        "protein": 5,
        "carbs": 20,
        "fat": 3,
        "fiber": 2,
        "sugar": 1
    }
    
    multiplier = get_quantity_multiplier(quantity, unit)
    for key in base_nutrition:
        base_nutrition[key] = round(base_nutrition[key] * multiplier, 1)
    
    return base_nutrition
