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



@csrf_exempt
def api_login(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        return JsonResponse({
            "success": True, 
            "username": user.username,
            "has_profile": hasattr(user, 'userprofile')
        })
    else:
        return JsonResponse({"error": "Invalid username or password"}, status=400)



@csrf_exempt
def api_signup(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    if not username or not password:
        return JsonResponse({"error": "Username and password are required"}, status=400)
        
    from django.contrib.auth.models import User
    if User.objects.filter(username=username).exists():
        return JsonResponse({"error": "Username already exists"}, status=400)
        
    try:
        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        return JsonResponse({
            "success": True, 
            "username": user.username,
            "has_profile": False
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)



@csrf_exempt
def api_logout(request):
    from django.contrib.auth import logout
    logout(request)
    return JsonResponse({"success": True})



def api_user_status(request):
    if request.user.is_authenticated:
        return JsonResponse({
            "authenticated": True,
            "username": request.user.username,
            "has_profile": hasattr(request.user, 'userprofile')
        })
    return JsonResponse({"authenticated": False})



@csrf_exempt
@login_required
def api_profile_get_or_post(request):
    try:
        profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        profile = None
        
    if request.method == 'GET':
        if not profile:
            return JsonResponse({"has_profile": False})
        return JsonResponse({
            "has_profile": True,
            "age": profile.age,
            "height": profile.height,
            "weight": profile.weight,
            "gender": profile.gender,
            "fitness_level": profile.fitness_level,
            "primary_goal": profile.primary_goal,
            "injuries_or_limitations": profile.injuries_or_limitations,
            "available_time": profile.available_time,
            "weak_muscles": profile.weak_muscles,
            "equipment_available": profile.equipment_available,
            "calories_per_day": profile.calories_per_day,
            "bmi": profile.bmi,
            "bmi_category": profile.bmi_category
        })
        
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
            
        age = data.get('age')
        height = data.get('height')
        weight = data.get('weight')
        gender = data.get('gender')
        fitness_level = data.get('fitness_level')
        primary_goal = data.get('primary_goal')
        available_time = data.get('available_time')
        
        if not all([age, height, weight, gender, fitness_level, primary_goal, available_time]):
            return JsonResponse({"error": "Missing required fields"}, status=400)
            
        if not profile:
            profile = UserProfile(user=request.user)
            
        profile.age = int(age)
        profile.height = float(height)
        profile.weight = float(weight)
        profile.gender = gender
        profile.fitness_level = fitness_level
        profile.primary_goal = primary_goal
        profile.injuries_or_limitations = data.get('injuries_or_limitations', '')
        profile.available_time = int(available_time)
        profile.weak_muscles = data.get('weak_muscles', '')
        profile.equipment_available = data.get('equipment_available', '')
        profile.calories_per_day = data.get('calories_per_day')
        profile.save()
        
        # Clear existing chat session
        session_id = request.session.get('chat_session_id')
        if session_id:
            try:
                ChatSession.objects.filter(session_id=session_id, user=request.user).delete()
                del request.session['chat_session_id']
            except Exception:
                pass
                
        return JsonResponse({"success": True})
        
    return JsonResponse({"error": "Method not allowed"}, status=405)


