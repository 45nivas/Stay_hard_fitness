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
def fitness_chat(request):
    """Fitness chatbot interface with intent-based routing and multi-tier fallback"""
    from workouts.chat.classifier import classify_intent
    from workouts.chat.engine import get_chat_response
    from workouts.chat.cache import get_cached_response, set_cached_response

    # Get or create chat session
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={'user': request.user}
    )
    
    # Initialize chatbot for profile summary helper
    from workouts.fitness_chatbot import FitnessChatbot
    chatbot = FitnessChatbot()
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        chatbot.user_data = {
            'height': user_profile.height,
            'weight': user_profile.weight,
            'age': user_profile.age,
            'gender': user_profile.get_gender_display(),
            'fitness_level': user_profile.get_fitness_level_display(),
            'goals': [user_profile.get_primary_goal_display()],
            'primary_goal': user_profile.primary_goal,
            'injuries_or_limitations': user_profile.injuries_or_limitations,
            'available_time': user_profile.available_time,
            'weak_muscles': user_profile.weak_muscles.split(',') if user_profile.weak_muscles else [],
            'equipment_available': user_profile.equipment_available.split(',') if user_profile.equipment_available else [],
            'calories_per_day': user_profile.calories_per_day,
        }
    except UserProfile.DoesNotExist:
        if session.user_data:
            chatbot.user_data = session.user_data

    if request.method == 'POST':
        user_message = request.POST.get("message", "").strip()
        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)
            
        # Layer 1: classify
        intent = classify_intent(user_message)
        
        # Layer 2: check cache
        cached = get_cached_response(intent, user_message)
        if cached:
            ChatMessage.objects.create(
                session=session,
                message=user_message,
                response=cached
            )
            return JsonResponse({
                "success": True,
                "response": cached,
                "reply": cached, 
                "intent": intent, 
                "tier": "cache"
            })
            
        # Context-aware enhancement: inject user profile to user message for accurate calculations
        context_msg = user_message
        try:
            user_profile = UserProfile.objects.get(user=request.user)
            profile_context = f"[Context: User weight={user_profile.weight}kg, height={user_profile.height}cm, age={user_profile.age}, gender={user_profile.get_gender_display()}, goal={user_profile.get_primary_goal_display()}]"
            context_msg = f"{profile_context} {user_message}"
        except UserProfile.DoesNotExist:
            pass

        # Layer 3: get response through fallback chain
        result = get_chat_response(intent, context_msg)
        bot_response = result["reply"]
        
        # Save chat message in database
        ChatMessage.objects.create(
            session=session,
            message=user_message,
            response=bot_response
        )
        
        # Layer 4: cache the result (use user_message as key)
        set_cached_response(intent, user_message, bot_response)
        
        return JsonResponse({
            "success": True,
            "response": bot_response,
            "reply": bot_response,
            "intent": intent,
            "tier": result["tier"]
        })
        
    # GET request
    form = ChatMessageForm()
    show_history = request.GET.get('show_history', 'false') == 'true'
    if show_history:
        messages = session.messages.all().order_by('-timestamp')[:5]
    else:
        messages = []
        
    if not messages and not show_history:
        welcome_message = "Welcome to OS Architect. I am your Senior Fitness & Nutrition Coach. Let's build your transformation protocol or address your biomechanics queries."
    else:
        welcome_message = None
        
    context = {
        'form': form,
        'messages': messages,
        'welcome_message': welcome_message,
        'user_data': chatbot.get_user_profile_summary(),
        'session_id': session_id,
        'is_gemini_active': bool(os.getenv("GEMINI_API_KEY"))
    }
    
    return render(request, 'fitness_chat.html', context)



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


@csrf_exempt
@login_required
def api_fitness_chat(request):
    from workouts.chat.classifier import classify_intent
    from workouts.chat.engine import get_chat_response
    from workouts.chat.cache import get_cached_response, set_cached_response
    
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
        
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={'user': request.user}
    )
    
    chatbot = FitnessChatbot()
    try:
        user_profile = UserProfile.objects.get(user=request.user)
        chatbot.user_data = {
            'height': user_profile.height,
            'weight': user_profile.weight,
            'age': user_profile.age,
            'gender': user_profile.get_gender_display(),
            'fitness_level': user_profile.get_fitness_level_display(),
            'goals': [user_profile.get_primary_goal_display()],
            'primary_goal': user_profile.primary_goal,
            'injuries_or_limitations': user_profile.injuries_or_limitations,
            'available_time': user_profile.available_time,
            'weak_muscles': user_profile.weak_muscles.split(',') if user_profile.weak_muscles else [],
            'equipment_available': user_profile.equipment_available.split(',') if user_profile.equipment_available else [],
            'calories_per_day': user_profile.calories_per_day,
        }
    except UserProfile.DoesNotExist:
        if session.user_data:
            chatbot.user_data = session.user_data
            
    if request.method == 'GET':
        messages = session.messages.all().order_by('timestamp')
        messages_list = []
        for m in messages:
            messages_list.append({
                "message": m.message,
                "response": m.response,
                "timestamp": m.timestamp.isoformat()
            })
            
        welcome_message = "Welcome to OS Architect. I am your Senior Fitness & Nutrition Coach. Let's build your transformation protocol or address your biomechanics queries."
        
        return JsonResponse({
            "messages": messages_list,
            "welcome_message": welcome_message if not messages_list else None,
            "user_summary": chatbot.get_user_profile_summary(),
            "is_gemini_active": bool(os.getenv("GEMINI_API_KEY"))
        })
        
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "").strip()
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
            
        if not user_message:
            return JsonResponse({"error": "Empty message"}, status=400)
            
        intent = classify_intent(user_message)
        cached = get_cached_response(intent, user_message)
        if cached:
            ChatMessage.objects.create(
                session=session,
                message=user_message,
                response=cached
            )
            return JsonResponse({
                "success": True,
                "response": cached,
                "reply": cached,
                "intent": intent,
                "tier": "cache"
            })
            
        context_msg = user_message
        try:
            user_profile = UserProfile.objects.get(user=request.user)
            profile_context = f"[Context: User weight={user_profile.weight}kg, height={user_profile.height}cm, age={user_profile.age}, gender={user_profile.get_gender_display()}, goal={user_profile.get_primary_goal_display()}]"
            context_msg = f"{profile_context} {user_message}"
        except UserProfile.DoesNotExist:
            pass
            
        result = get_chat_response(intent, context_msg)
        bot_response = result["reply"]
        
        ChatMessage.objects.create(
            session=session,
            message=user_message,
            response=bot_response
        )
        
        set_cached_response(intent, user_message, bot_response)
        
        return JsonResponse({
            "success": True,
            "response": bot_response,
            "reply": bot_response,
            "intent": intent,
            "tier": result["tier"]
        })
        
    return JsonResponse({"error": "Method not allowed"}, status=405)


