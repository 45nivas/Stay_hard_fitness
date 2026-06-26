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
from workouts.models import FoodItem, MealLog, DailySummary

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

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


@login_required
def analytics_dashboard(request):
    """Fitness Analytics Dashboard displaying logs, streaks, and muscle volume."""
    user = request.user
    
    if request.method == 'POST':
        exercise_name = request.POST.get('exercise_name', '').strip()
        sets = int(request.POST.get('sets', 1))
        reps = int(request.POST.get('reps', 0))
        weight = float(request.POST.get('weight', 0.0))
        muscle_group = request.POST.get('muscle_group', 'General').strip()
        
        if exercise_name and reps > 0:
            from .shared import get_exercise_id_by_name, get_exercise_by_id, normalize_exercise_name
            normalized_name = normalize_exercise_name(exercise_name)
            exercise_id = get_exercise_id_by_name(normalized_name)
            if exercise_id:
                details = get_exercise_by_id(exercise_id)
                if details:
                    exercise_name = details["name"]
                    muscle_group = details["muscle_group"]

            WorkoutLog.objects.create(
                user=user,
                exercise_name=exercise_name,
                exercise_id=exercise_id,
                sets=sets,
                reps=reps,
                weight=weight,
                muscle_group=muscle_group,
                duration_minutes=sets * 2
            )
            messages.success(request, f"Logged {exercise_name} successfully!")
            return redirect('analytics')
            
    # Calculate Streak
    logs_dates = WorkoutLog.objects.filter(user=user).values_list('date', flat=True).distinct().order_by('-date')
    streak = 0
    if logs_dates:
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        most_recent = logs_dates[0]
        if most_recent == today or most_recent == yesterday:
            streak = 0
            expected_date = most_recent
            for log_date in logs_dates:
                if log_date == expected_date:
                    streak += 1
                    expected_date -= datetime.timedelta(days=1)
                elif log_date > expected_date:
                    continue
                else:
                    break

    # Calculations for Volume by Muscle Group
    logs = WorkoutLog.objects.filter(user=user)
    total_workouts = logs.count()
    total_volume_lifted = sum(log.total_volume for log in logs)
    total_active_minutes = logs.aggregate(Sum('duration_minutes'))['duration_minutes__sum'] or 0
    
    # Posture Analytics Average
    avg_posture = PostureAnalysis.objects.filter(user=user).aggregate(Avg('posture_score'))['posture_score__avg'] or 0.0
    avg_posture = round(avg_posture, 1)

    # Muscle Volume Breakdown for Radar Chart
    muscle_groups = ['Legs', 'Chest', 'Biceps', 'Shoulders', 'Back']
    muscle_volume_data = {mg: 0.0 for mg in muscle_groups}
    for log in logs:
        mg = log.muscle_group
        if mg not in muscle_volume_data:
            muscle_volume_data[mg] = 0.0
        muscle_volume_data[mg] += log.total_volume

    # Monthly Progress Trend for Line Chart (Last 6 Months)
    six_months_ago = datetime.date.today() - datetime.timedelta(days=180)
    recent_logs = WorkoutLog.objects.filter(user=user, date__gte=six_months_ago).order_by('date')
    
    monthly_volume_data = {}
    for log in recent_logs:
        month_str = log.date.strftime('%b')
        monthly_volume_data[month_str] = monthly_volume_data.get(month_str, 0.0) + log.total_volume

    # Ensure last 6 months are present in order
    ordered_months = []
    current_date = datetime.date.today()
    for i in range(5, -1, -1):
        check_date = current_date - datetime.timedelta(days=i*30)
        ordered_months.append(check_date.strftime('%b'))

    ordered_monthly_values = [round(monthly_volume_data.get(m, 0.0), 1) for m in ordered_months]

    # Fetch last 5 recent workout logs for history table
    recent_logs_list = logs.order_by('-date')[:5]

    # Fetch latest WorkoutRecommendation if it exists
    from workouts.models import WorkoutRecommendation
    recommendation = WorkoutRecommendation.objects.filter(user_profile__user=user).order_by('-created_at').first()

    context = {
        'streak': streak,
        'total_workouts': total_workouts,
        'total_volume_lifted': round(total_volume_lifted, 1),
        'total_active_minutes': total_active_minutes,
        'avg_posture': avg_posture,
        'recent_logs': recent_logs_list,
        'recommendation': recommendation,
        # JSON data for charts
        'muscle_labels': json.dumps(list(muscle_volume_data.keys())),
        'muscle_values': json.dumps(list(muscle_volume_data.values())),
        'trend_labels': json.dumps(ordered_months),
        'trend_values': json.dumps(ordered_monthly_values),
    }
    
    return render(request, 'analytics.html', context)



@csrf_exempt
@login_required
def api_one_rep_max_calculator(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        data = json.loads(request.body)
        weight = float(data.get('weight', 0))
        reps = int(data.get('reps', 0))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
        
    if weight <= 0 or reps <= 0:
        return JsonResponse({"error": "Weight and reps must be positive numbers"}, status=400)
        
    epley = weight * (1 + reps / 30.0)
    if reps < 37:
        brzycki = weight / (1.0278 - (0.0278 * reps))
    else:
        brzycki = epley
        
    lander = (100.0 * weight) / (101.3 - 2.6712 * reps)
    
    one_rep_max = round((epley + brzycki + lander) / 3.0, 1)
    
    percentages = {}
    for pct in range(50, 105, 5):
        percentages[f"{pct}%"] = round(one_rep_max * (pct / 100.0), 1)
        
    return JsonResponse({
        "one_rep_max": one_rep_max,
        "percentages": percentages,
        "epley": round(epley, 1),
        "brzycki": round(brzycki, 1),
        "lander": round(lander, 1)
    })



@csrf_exempt
@login_required
def api_carb_cycling_calculator(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)
    try:
        data = json.loads(request.body)
        age = int(data.get('age', 0))
        gender = data.get('gender', 'male')
        height = float(data.get('height', 0))
        weight = float(data.get('weight', 0))
        height_unit = data.get('height_unit', 'cm')
        weight_unit = data.get('weight_unit', 'kg')
        activity_level = data.get('activity_level', 'moderate')
        goal = data.get('goal', 'maintenance')
        training_days = int(data.get('training_days', 3))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
        
    if height_unit == 'ft':
        height = height * 30.48
    if weight_unit == 'lbs':
        weight = weight * 0.453592
        
    if gender == 'male':
        bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) + 5.0
    else:
        bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age) - 161.0
        
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    
    multiplier = activity_multipliers.get(activity_level, 1.2)
    tdee = bmr * multiplier
    
    if goal == 'weight_loss':
        target_calories = tdee - 500
    elif goal == 'muscle_gain':
        target_calories = tdee + 400
    else:
        target_calories = tdee
        
    protein_g = round(weight * 2.2)
    protein_cal = protein_g * 4
    
    high_carb_cal = target_calories + 200
    high_fat_cal = high_carb_cal * 0.25
    high_fat_g = round(high_fat_cal / 9.0)
    high_carb_cal_rem = high_carb_cal - (protein_cal + high_fat_cal)
    high_carb_g = round(max(high_carb_cal_rem / 4.0, 0))
    
    low_carb_cal = target_calories - 300
    low_fat_cal = low_carb_cal * 0.35
    low_fat_g = round(low_fat_cal / 9.0)
    low_carb_cal_rem = low_carb_cal - (protein_cal + low_fat_cal)
    low_carb_g = round(max(low_carb_cal_rem / 4.0, 0))
    
    return JsonResponse({
        "bmr": round(bmr),
        "tdee": round(tdee),
        "target_calories": round(target_calories),
        "protein_g": protein_g,
        "high_carb_day": {
            "calories": round(high_carb_cal),
            "carbs_g": high_carb_g,
            "protein_g": protein_g,
            "fat_g": high_fat_g
        },
        "low_carb_day": {
            "calories": round(low_carb_cal),
            "carbs_g": low_carb_g,
            "protein_g": protein_g,
            "fat_g": low_fat_g
        }
    })


