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

def smart_food_lookup(food_name, quantity, unit):
    """
    Enhanced food lookup with the following fallback chain:
    Indian staples dict (rapidfuzz >= 80) -> NUTRITION_DATABASE -> USDA API -> Gemini -> Hardcoded defaults.
    Always returns values normalized per 100g, and source name.
    """
    food_lower = food_name.lower().strip()
    multiplier = get_quantity_multiplier(quantity, unit)
    if multiplier <= 0:
        multiplier = 1.0
        
    # 1. Check Indian staples dictionary first using fuzzy matching (rapidfuzz)
    try:
        from workouts.indian_foods import INDIAN_FOODS_DB
        from rapidfuzz import process, fuzz
        choices = list(INDIAN_FOODS_DB.keys())
        match_result = process.extractOne(food_lower, choices, scorer=fuzz.token_sort_ratio)
        if match_result:
            matched_name, score, index = match_result
            if score >= 80:
                print(f"DEBUG: Indian Foods DB match found: {matched_name} (score: {score})")
                return INDIAN_FOODS_DB[matched_name], "Indian Foods DB"
    except Exception as e:
        print(f"Error in Indian Foods DB fuzzy match: {e}")

    # 2. Check original NUTRITION_DATABASE
    # Exact match
    if food_lower in NUTRITION_DATABASE:
        return NUTRITION_DATABASE[food_lower], "Database"
    # Fuzzy match
    for key, data in NUTRITION_DATABASE.items():
        food_words = food_lower.split()
        if any(word in key for word in food_words if len(word) > 3) or \
           any(word in food_lower for word in key.split() if len(word) > 3):
            return data, "Database"
            
    if 'egg' in food_lower:
        if 'white' in food_lower:
            return NUTRITION_DATABASE['egg white'], "Database"
        else:
            return NUTRITION_DATABASE['whole egg'], "Database"

    # 3. Query USDA FoodData Central API
    usda_key = os.getenv("USDA_API_KEY")
    if usda_key:
        print(f"DEBUG: Querying USDA API for: {food_lower}")
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            'query': food_lower,
            'api_key': usda_key,
            'pageSize': 3
        }
        try:
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                resp_data = resp.json()
                foods = resp_data.get('foods', [])
                if foods:
                    first_food = foods[0]
                    nutrients = first_food.get('foodNutrients', [])
                    
                    id_mapping = {
                        1008: 'calories',
                        1003: 'protein',
                        1005: 'carbs',
                        1004: 'fat',
                        1079: 'fiber',
                        1093: 'sodium'
                    }
                    
                    usda_per_100g = {
                        'calories': 0.0,
                        'protein': 0.0,
                        'carbs': 0.0,
                        'fat': 0.0,
                        'fiber': 0.0,
                        'sodium': 0.0
                    }
                    
                    for nut in nutrients:
                        n_id = nut.get('nutrientId')
                        if n_id in id_mapping:
                            key = id_mapping[n_id]
                            val = float(nut.get('value', 0))
                            if n_id == 1093:
                                # Sodium is in mg in USDA, convert to grams
                                usda_per_100g[key] = val / 1000.0
                            else:
                                usda_per_100g[key] = val
                                
                    print(f"DEBUG: USDA match found: {first_food.get('description')} (Cals per 100g: {usda_per_100g['calories']})")
                    return usda_per_100g, "USDA API"
        except Exception as e:
            print(f"USDA API error: {e}")

    # 4. Query Gemini API
    if GEMINI_API_KEY:
        print(f"DEBUG: Querying Gemini for: {food_lower}")
        try:
            gemini_data = query_gemini_nutrition(food_name, quantity, unit)
            # gemini_data is already scaled to quantity. Convert back to per-100g base for scaling consistency
            usda_per_100g = {
                'calories': round(float(gemini_data.get('calories', 0)) / multiplier, 1),
                'protein': round(float(gemini_data.get('protein', 0)) / multiplier, 1),
                'carbs': round(float(gemini_data.get('carbs', 0)) / multiplier, 1),
                'fat': round(float(gemini_data.get('fat', 0)) / multiplier, 1),
                'fiber': round(float(gemini_data.get('fiber', 0)) / multiplier, 1),
                'sodium': round(float(gemini_data.get('sodium', 0)) / multiplier, 3)
            }
            return usda_per_100g, "Gemini AI"
        except Exception as e:
            print(f"Gemini API error during fallback: {e}")

    # 5. Hardcoded defaults (400 kcal, 20g protein, 45g carbs, 15g fat, 5g fiber per 100g)
    print("DEBUG: Using hardcoded fallback defaults")
    defaults_per_100g = {
        'calories': 400.0,
        'protein': 20.0,
        'carbs': 45.0,
        'fat': 15.0,
        'fiber': 5.0,
        'sodium': 0.0
    }
    return defaults_per_100g, "Hardcoded Defaults"



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
                        
                        if db_data:
                            # Use our corrected database or API values from our structured fallback chain
                            multiplier = get_quantity_multiplier(quantity, unit)
                            print(f"DEBUG: Using {source} - Multiplier={multiplier}, Final calories={round(db_data['calories'] * multiplier, 1)}")
                            
                            enhanced_food = {
                                'food': food.get('food'),
                                'quantity': quantity,
                                'unit': unit,
                                'calories': round(db_data['calories'] * multiplier, 1),
                                'protein': round(db_data['protein'] * multiplier, 1),
                                'carbs': round(db_data['carbs'] * multiplier, 1),
                                'fat': round(db_data['fat'] * multiplier, 1),
                                'fiber': round(db_data['fiber'] * multiplier, 1),
                                'sugar': round(db_data.get('sugar', 0) * multiplier, 1),
                                'sodium': round(db_data.get('sodium', 0) * multiplier, 3),
                                'source': source
                            }
                            print(f"DEBUG: {source} result: {enhanced_food['food']} -> {enhanced_food['calories']} cal")
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
        meals = MealLog.objects.filter(user=request.user, date=today)
        
        updated_count = 0
        for meal in meals:
            # Recalculate nutrition using current database
            db_data, source = smart_food_lookup(meal.food_item.name, meal.quantity, meal.unit)
            
            if db_data:
                # Recalculate with correct values
                multiplier = get_quantity_multiplier(meal.quantity, meal.unit)
                
                meal.calories = round(db_data['calories'] * multiplier, 1)
                meal.protein = round(db_data['protein'] * multiplier, 1)
                meal.carbs = round(db_data['carbs'] * multiplier, 1)
                meal.fat = round(db_data['fat'] * multiplier, 1)
                meal.fiber = round(db_data['fiber'] * multiplier, 1)
                meal.sugar = round(db_data.get('sugar', 0) * multiplier, 1)
                meal.sodium = round(db_data.get('sodium', 0) * multiplier, 3)
                meal.source = f'{source} (Updated)'
                
                meal.save()
                updated_count += 1
                print(f"DEBUG: Updated {meal.food_item.name} - {meal.quantity}{meal.unit} -> {meal.calories} calories")
        
        return JsonResponse({
            "success": True,
            "message": f"Updated {updated_count} meals with corrected nutrition data",
            "updated_count": updated_count
        })
        
    except Exception as e:
        return JsonResponse({"error": f"Error recalculating meals: {str(e)}"}, status=500)


COMPOUND_FOODS = {
    "chicken breast": "chicken",
    "brown rice": "rice",
    "white rice": "rice",
    "whole egg": "egg",
    "rolled oats": "oats",
    "sweet potato": "potato",
}


def normalize_query(food_name):
    lower = food_name.lower().strip()
    if lower in COMPOUND_FOODS:
        return COMPOUND_FOODS[lower]
    words = lower.split()
    return words[-1] if words else lower


def get_top_candidates(food_query, quantity, unit):
    candidates = []
    food_lower = food_query.lower().strip()
    
    # 1. Check Indian Foods DB fuzzy matching
    try:
        from workouts.indian_foods import INDIAN_FOODS_DB
        from rapidfuzz import process, fuzz
        choices = list(INDIAN_FOODS_DB.keys())
        matches = process.extract(food_lower, choices, scorer=fuzz.token_sort_ratio, limit=3)
        for matched_name, score, index in matches:
            if score >= 60:
                candidates.append({
                    'name': matched_name.title(),
                    'data': INDIAN_FOODS_DB[matched_name],
                    'source': 'Indian Foods DB'
                })
    except Exception as e:
        print(f"Error in Indian Foods candidate extraction: {e}")

    # 2. Check NUTRITION_DATABASE exact or fuzzy
    for key, data in NUTRITION_DATABASE.items():
        if food_lower in key or key in food_lower:
            if not any(c['name'].lower() == key.lower() for c in candidates):
                candidates.append({
                    'name': key.title(),
                    'data': data,
                    'source': 'Database'
                })

    # 3. Query USDA API for brands/alternatives
    usda_key = os.getenv("USDA_API_KEY")
    if usda_key and len(candidates) < 3:
        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            'query': food_lower,
            'api_key': usda_key,
            'pageSize': 5
        }
        try:
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                resp_data = resp.json()
                for first_food in resp_data.get('foods', []):
                    desc = first_food.get('description', '')
                    if any(c['name'].lower() == desc.lower() for c in candidates):
                        continue
                    
                    nutrients = first_food.get('foodNutrients', [])
                    id_mapping = {
                        1008: 'calories',
                        1003: 'protein',
                        1005: 'carbs',
                        1004: 'fat',
                        1079: 'fiber',
                        1093: 'sodium'
                    }
                    usda_per_100g = {'calories': 0.0, 'protein': 0.0, 'carbs': 0.0, 'fat': 0.0, 'fiber': 0.0, 'sodium': 0.0}
                    for nut in nutrients:
                        n_id = nut.get('nutrientId')
                        if n_id in id_mapping:
                            key = id_mapping[n_id]
                            val = float(nut.get('value', 0))
                            if n_id == 1093:
                                usda_per_100g[key] = val / 1000.0
                            else:
                                usda_per_100g[key] = val
                    
                    candidates.append({
                        'name': desc.title(),
                        'data': usda_per_100g,
                        'source': 'USDA API'
                    })
                    if len(candidates) >= 5:
                        break
        except Exception as e:
            print(f"USDA candidate extraction error: {e}")

    # If empty, run fallback smart_food_lookup
    if not candidates:
        db_data, source = smart_food_lookup(food_query, quantity, unit)
        candidates.append({
            'name': food_query.title(),
            'data': db_data,
            'source': source
        })

    # Fill to exactly 3 if needed
    if len(candidates) < 3:
        generic_data = candidates[0]['data']
        source = candidates[0]['source']
        name = candidates[0]['name']
        candidates.append({
            'name': f"Organic {name}",
            'data': generic_data,
            'source': source
        })
        candidates.append({
            'name': f"Generic {name}",
            'data': generic_data,
            'source': source
        })

    # Limit to top 3 and calculate scaled calories/macros
    top_3 = candidates[:3]
    multiplier = get_quantity_multiplier(quantity, unit)
    if multiplier <= 0:
        multiplier = 1.0
        
    formatted = []
    for c in top_3:
        data = c['data']
        formatted.append({
            'name': c['name'],
            'source': c['source'],
            'calories': round(data['calories'] * multiplier, 1),
            'protein': round(data['protein'] * multiplier, 1),
            'carbs': round(data['carbs'] * multiplier, 1),
            'fat': round(data['fat'] * multiplier, 1),
            'fiber': round(data.get('fiber', 0) * multiplier, 1),
            'sodium': round(data.get('sodium', 0) * multiplier, 3),
        })
    return formatted



def update_daily_summary(user, date):
    from django.db.models import Sum
    from decimal import Decimal
    
    logs = MealLog.objects.filter(user=user, date=date)
    
    total_cal = sum(log.calories for log in logs)
    total_pro = sum(log.protein for log in logs)
    total_carb = sum(log.carbs for log in logs)
    total_fat = sum(log.fat for log in logs)
    total_fib = sum(log.fiber for log in logs)
    total_sug = sum(log.sugar for log in logs)
    total_sod = sum(log.sodium for log in logs)
    
    summary, created = DailySummary.objects.get_or_create(
        user=user,
        date=date
    )
    summary.total_calories = int(round(total_cal))
    summary.total_protein = Decimal(str(round(total_pro, 2)))
    summary.total_carbohydrates = Decimal(str(round(total_carb, 2)))
    summary.total_fats = Decimal(str(round(total_fat, 2)))
    summary.total_fiber = Decimal(str(round(total_fib, 2)))
    summary.total_sugar = Decimal(str(round(total_sug, 2)))
    summary.total_sodium = Decimal(str(round(total_sod, 3)))
    summary.save()



@csrf_exempt
@login_required
def log_meal_from_voice(request):
    """Simple but accurate nutrition tracking with adaptive food preferences"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        data = json.loads(request.body)
        transcribed_text = data.get("text") or data.get("transcript", "")
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    if not transcribed_text:
        return JsonResponse({"error": "No transcribed text provided"}, status=400)

    # Parse food item candidates/names from raw speech text
    try:
        foods = query_gemini_for_foods(transcribed_text)
    except Exception as e:
        return JsonResponse({"error": f"Error analyzing food: {str(e)}"}, status=500)
    
    if not isinstance(foods, list) or not foods:
        return JsonResponse({"error": "Could not understand the food. Try: '2 eggs and 100g oats'"}, status=400)

    results = []
    today = timezone.now().date()

    for idx, food_obj in enumerate(foods):
        food = food_obj.get("food")
        quantity = float(food_obj.get("quantity", 1))
        unit = food_obj.get("unit", "")
        
        if not food:
            continue

        normalized = normalize_query(food)
        pref = FoodPreference.objects.filter(user=request.user, food_query=normalized).first()

        if pref and pref.log_count >= 3:
            # Auto-log mode!
            # Use preferred_food_data scaled to spoken quantity
            multiplier = get_quantity_multiplier(quantity, unit)
            if multiplier <= 0:
                multiplier = 1.0

            calories = round(pref.preferred_food_data['calories'] * multiplier, 1)
            protein = round(pref.preferred_food_data['protein'] * multiplier, 1)
            carbs = round(pref.preferred_food_data['carbs'] * multiplier, 1)
            fat = round(pref.preferred_food_data['fat'] * multiplier, 1)
            fiber = round(pref.preferred_food_data.get('fiber', 0) * multiplier, 1)
            sodium = round(pref.preferred_food_data.get('sodium', 0) * multiplier, 3)

            # Find or create FoodItem
            food_item, _ = FoodItem.objects.get_or_create(
                name=pref.preferred_food_name.lower().strip(),
                defaults={
                    "calories_per_100g": pref.preferred_food_data['calories'],
                    "protein": pref.preferred_food_data['protein'],
                    "carbs": pref.preferred_food_data['carbs'],
                    "fat": pref.preferred_food_data['fat'],
                    "fiber": pref.preferred_food_data.get('fiber', 0),
                    "sodium": pref.preferred_food_data.get('sodium', 0),
                }
            )

            # Save MealLog
            meal = MealLog.objects.create(
                user=request.user,
                food_item=food_item,
                quantity=quantity,
                unit=unit,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
                fiber=fiber,
                sodium=sodium,
                source='Preferred',
                date=today
            )

            # Update DailySummary
            update_daily_summary(request.user, today)

            # Increment log count / last used
            pref.log_count += 1
            pref.save()

            results.append({
                "id": meal.id,
                "food": pref.preferred_food_name,
                "quantity": quantity,
                "unit": unit,
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat,
                "fiber": fiber,
                "sodium": sodium,
            })
        else:
            # Picker mode (needs confirmation)!
            # Run fallback chain to get top 3 candidates
            candidates = get_top_candidates(food, quantity, unit)
            
            preferred_name = pref.preferred_food_name if pref else candidates[0]['name']

            return JsonResponse({
                "success": True,
                "mode": "picker",
                "food_query": normalized,
                "original_query": food,
                "quantity": quantity,
                "unit": unit,
                "candidates": candidates,
                "preferred": preferred_name
            })

    # Get today's totals (if we successfully auto-logged everything)
    logs = MealLog.objects.filter(user=request.user, date=today)
    total_calories = sum(log.calories for log in logs)
    total_protein = sum(log.protein for log in logs)
    total_carbs = sum(log.carbs for log in logs)
    total_fats = sum(log.fat for log in logs)
    total_fiber = sum(log.fiber for log in logs)
    total_sugar = sum(log.sugar for log in logs)
    total_sodium = sum(log.sodium for log in logs)

    response_data = {
        "success": True,
        "mode": "auto",
        "logged": results,
        "date": str(today),
        "total_calories": total_calories,
        "total_protein": total_protein,
        "total_carbs": total_carbs,
        "total_fats": total_fats,
        "total_fiber": total_fiber,
        "total_sugar": total_sugar,
        "total_sodium": total_sodium,
    }
    if len(results) == 1:
        response_data["meal_id"] = results[0]["id"]
        
    return JsonResponse(response_data)



@csrf_exempt
@login_required
def confirm_meal(request):
    """Confirm and log picker meal choices, updating daily summary and food preferences"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        data = json.loads(request.body)
        food_query = data.get("food_query", "").lower().strip()
        chosen_food_name = data.get("chosen_food_name", "").strip()
        chosen_food_data = data.get("chosen_food_data", {})
        quantity = float(data.get("quantity", 100))
        unit = data.get("unit", "g")
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
        
    if not chosen_food_name or not chosen_food_data:
        return JsonResponse({"error": "Missing selection data"}, status=400)
        
    try:
        # Calculate base values per 100g base for preference scaling
        multiplier = get_quantity_multiplier(quantity, unit)
        if multiplier <= 0:
            multiplier = 1.0
            
        base_calories = float(chosen_food_data.get('calories', 0)) / multiplier
        base_protein = float(chosen_food_data.get('protein', 0)) / multiplier
        base_carbs = float(chosen_food_data.get('carbs', 0)) / multiplier
        base_fat = float(chosen_food_data.get('fat', 0)) / multiplier
        base_fiber = float(chosen_food_data.get('fiber', 0)) / multiplier
        base_sodium = float(chosen_food_data.get('sodium', 0)) / multiplier
        
        # Save or update FoodItem
        food_item, _ = FoodItem.objects.get_or_create(
            name=chosen_food_name.lower().strip(),
            defaults={
                "calories_per_100g": base_calories,
                "protein": base_protein,
                "carbs": base_carbs,
                "fat": base_fat,
                "fiber": base_fiber,
                "sodium": base_sodium,
            }
        )
        
        # Save MealLog
        today = timezone.now().date()
        meal = MealLog.objects.create(
            user=request.user,
            food_item=food_item,
            quantity=quantity,
            unit=unit,
            calories=float(chosen_food_data.get('calories', 0)),
            protein=float(chosen_food_data.get('protein', 0)),
            carbs=float(chosen_food_data.get('carbs', 0)),
            fat=float(chosen_food_data.get('fat', 0)),
            fiber=float(chosen_food_data.get('fiber', 0)),
            sodium=float(chosen_food_data.get('sodium', 0)),
            source='USDA',
            date=today
        )
        
        # Update DailySummary
        update_daily_summary(request.user, today)
        
        # Upsert FoodPreference
        normalized = normalize_query(food_query)
        pref, created = FoodPreference.objects.get_or_create(
            user=request.user,
            food_query=normalized,
            defaults={
                'preferred_food_name': chosen_food_name,
                'preferred_food_data': {
                    'calories': base_calories,
                    'protein': base_protein,
                    'carbs': base_carbs,
                    'fat': base_fat,
                    'fiber': base_fiber,
                    'sodium': base_sodium,
                },
                'log_count': 1
            }
        )
        if not created:
            pref.log_count += 1
            pref.preferred_food_name = chosen_food_name
            pref.preferred_food_data = {
                'calories': base_calories,
                'protein': base_protein,
                'carbs': base_carbs,
                'fat': base_fat,
                'fiber': base_fiber,
                'sodium': base_sodium,
            }
            pref.save()

        # Calculate today's totals
        logs = MealLog.objects.filter(user=request.user, date=today)
        total_calories = sum(log.calories for log in logs)
        total_protein = sum(log.protein for log in logs)
        total_carbs = sum(log.carbs for log in logs)
        total_fats = sum(log.fat for log in logs)
        total_fiber = sum(log.fiber for log in logs)
        total_sugar = sum(log.sugar for log in logs)
        total_sodium = sum(log.sodium for log in logs)

        return JsonResponse({
            "success": True,
            "logged": [{
                "food": chosen_food_name,
                "quantity": quantity,
                "unit": unit,
                "calories": meal.calories,
                "protein": meal.protein,
                "carbs": meal.carbs,
                "fat": meal.fat,
                "fiber": meal.fiber,
                "sodium": meal.sodium,
            }],
            "meal_id": meal.id,
            "date": str(today),
            "total_calories": total_calories,
            "total_protein": total_protein,
            "total_carbs": total_carbs,
            "total_fats": total_fats,
            "total_fiber": total_fiber,
            "total_sugar": total_sugar,
            "total_sodium": total_sodium,
        })
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)



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
            meal_date = meal_log.date
            meal_log.delete()
            
            # Recalculate daily summary for that date
            update_daily_summary(request.user, meal_date)
            
            return JsonResponse({"success": True})
        except MealLog.DoesNotExist:
            return JsonResponse({"success": False, "error": "Meal not found"})
        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})
    
    return JsonResponse({"error": "POST required"}, status=400)


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

