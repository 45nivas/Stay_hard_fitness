import sys
import os
import json
import django

# Setup Django environment
sys.path.append('c:\\Users\\matta\\OneDrive\\Desktop\\resume_projects\\gym')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gym_project.settings')
django.setup()

from django.contrib.auth.models import User
from django.core.cache import cache
from workouts.chat.classifier import classify_intent
from workouts.chat.cache import get_cached_response, set_cached_response
from workouts.chat.engine import get_chat_response
from workouts.models import ChatSession, ChatMessage

def run_verification():
    print("=== STARTING CHATBOT ROUTING VERIFICATION ===")
    
    # 1. Test Intent Classification
    test_queries = {
        "I have a sharp lower back pain after squats": "injury_pain",
        "give me a 4 day hypertrophy routine": "workout_plan",
        "what should i eat for dinner on a keto diet": "nutrition_plan",
        "calculate my tdee, I am 25 years old": "biometrics",
        "proper way to do a barbell bench press": "exercise_technique",
        "I feel like quitting the gym, no motivation": "motivation",
        "stuck at 80kg bench for 3 weeks": "plateau",
        "should I take creatine monohydrate?": "supplement",
        "who won the cricket match yesterday?": "off_topic",
        "what is progressive overload": "general_fitness"
    }
    
    classification_success = True
    for query, expected_intent in test_queries.items():
        intent = classify_intent(query)
        if intent == expected_intent:
            print(f"[OK] '{query}' -> {intent} (Expected: {expected_intent})")
        else:
            print(f"[FAIL] '{query}' -> {intent} (Expected: {expected_intent})")
            classification_success = False
            
    # 2. Test Response Caching Layer
    cache.clear()
    intent = "supplement"
    msg = "should I take creatine monohydrate?"
    
    # Verify cache miss
    cached = get_cached_response(intent, msg)
    if cached is None:
        print("[OK] Cache miss verified for new query")
    else:
        print("[FAIL] Cache hit on empty cache!")
        
    # Cache a mock response
    mock_resp = "OS Architect supplement advice: take 5g creatine."
    set_cached_response(intent, msg, mock_resp)
    
    # Verify cache hit
    cached = get_cached_response(intent, msg)
    if cached == mock_resp:
        print("[OK] Cache hit verified with correct response content")
    else:
        print(f"[FAIL] Cache hit failed: {cached}")
        
    # 3. Test Fallback Engine (Offline Templates)
    res = get_chat_response("injury_pain", "my shoulder hurts")
    print(f"Fallback response tier: {res['tier']}")
    print(f"Fallback reply: {res['reply'][:60]}...")
    
    if res['tier'] in ['offline', 'local_ollama_qwen']:
        print("[OK] Multi-tier fallback resolved correctly when cloud APIs are bypassed")
    else:
        print(f"[FAIL] Fallback resolved to unexpected tier: {res['tier']}")

    # 4. Test View Logic integration (AJAX simulation)
    from django.test import RequestFactory
    from workouts.views import fitness_chat
    
    user, _ = User.objects.get_or_create(username='test_chat_user')
    factory = RequestFactory()
    
    # Simulate first POST (should go through engine and write to cache)
    request = factory.post('/fitness-chat/', {'message': 'how do I fix squat form'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
    request.user = user
    request.session = {}
    
    response = fitness_chat(request)
    resp_data = json.loads(response.content)
    
    print(f"View Response data: {resp_data}")
    if resp_data.get('success') and resp_data.get('intent') == 'exercise_technique':
        print("[OK] View POST processed correctly (success, intent, tier keys present)")
    else:
        print("[FAIL] View POST response verification failed")
        
    # Simulate second POST (should hit cache)
    request2 = factory.post('/fitness-chat/', {'message': 'how do I fix squat form'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
    request2.user = user
    # transfer session data
    request2.session = request.session
    
    response2 = fitness_chat(request2)
    resp_data2 = json.loads(response2.content)
    print(f"Cached View Response data: {resp_data2}")
    
    if resp_data2.get('tier') == 'cache':
        print("[OK] Caching integration working end-to-end (second query resolved via cache)")
    else:
        print("[FAIL] Caching integration failed")
        
    print("=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    run_verification()
