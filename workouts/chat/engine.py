import os
import requests

OFFLINE_TEMPLATES = {
    "general_fitness": """OS Architect is currently operating in offline mode. Core directive: train with progressive overload, eat in alignment with your goal calories, sleep 7-9 hours, and stay consistent. Reconnect for personalized guidance.""",
    "injury_pain": """OS Architect offline. If you are in pain: stop training the affected area immediately. Rest, ice, compress, elevate. Consult a physiotherapist before resuming.""",
    "workout_plan": """OS Architect offline. Standard recommendations require profile configuration. Focus on a balanced push-pull-legs (PPL) or upper-lower split, tracking progress over time.""",
    "nutrition_plan": """OS Architect offline. Ensure a daily protein target of 1.6-2.2g per kg of bodyweight, whole foods, and a caloric layout suited to your primary goal (surplus for muscle gain, deficit for fat loss).""",
    "biometrics": """OS Architect offline. BMR = (10 × weight_kg) + (6.25 × height_cm) - (5 × age) [+5 for Men / -161 for Women]. Complete your profile setup for automated calculations.""",
    "exercise_technique": """OS Architect offline. Prioritize structural form: keep spine neutral, control the eccentric lowering phase, and execute through a full biomechanical range of motion.""",
    "motivation": """OS Architect offline. Motivation is secondary to consistency. Establish minimum effective dose routines, audit your recovery quality, and focus on today's session.""",
    "plateau": """OS Architect offline. A plateau is solved by tracking: verify calorie intake, ensure sufficient sleep/recovery, and introduce progressive overload adjustments.""",
    "supplement": """OS Architect offline. Tier 1 recommendations: Creatine monohydrate 5g/day, Caffeine 3-6mg/kg pre-workout. Food-first nutrition remains the primary driver of results.""",
    "off_topic": """SYSTEM ALERT: Query outside fitness domain.

OS Architect operates exclusively within:
→ Training & Programming
→ Nutrition & Macros
→ Biomechanics & Form
→ Recovery & Injury Prevention
→ Supplementation

Redirect your query to one of the above domains.
How can I optimize your performance today?"""
}

def get_groq_response(system_prompt, user_message, model):
    # Cloud API is commented out for testing as requested
    """
    try:
        from groq import Groq
        from django.conf import settings
        if not settings.GROQ_API_KEY:
            return None
        client = Groq(api_key=settings.GROQ_API_KEY)
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1024,
            temperature=0.7,
            timeout=8
        )
        return r.choices[0].message.content
    except Exception:
        return None
    """
    return None

def get_gemini_response(system_prompt, user_message):
    # Cloud API is commented out for testing as requested
    """
    import os
    import requests
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {'Content-Type': 'application/json'}
    prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_message}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        r = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=8)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception:
        pass
    """
    return None

def get_local_ollama_response(system_prompt, user_message, model="qwen2.5:3b"):
    """Local LLM tier for testing using lightweight (<3b parameter) models"""
    try:
        prompt = f"[SYSTEM] {system_prompt}\n\n[USER] {user_message}"
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512
                }
            },
            timeout=8
        )
        if r.status_code == 200:
            response_text = r.json().get("response", "").strip()
            if response_text:
                return response_text
    except Exception:
        pass
    return None

def get_chat_response(intent: str, user_message: str) -> dict:
    from workouts.chat.flows import FLOW_PROMPTS
    system_prompt = FLOW_PROMPTS.get(intent, FLOW_PROMPTS["general_fitness"])
    
    # Tier 1: Groq llama3-8b (Bypassed)
    response = get_groq_response(system_prompt, user_message, "llama3-8b-8192")
    if response:
        return {"reply": response, "tier": "groq_llama3", "intent": intent}
    
    # Tier 2: Groq mixtral (Bypassed)
    response = get_groq_response(system_prompt, user_message, "mixtral-8x7b-32768")
    if response:
        return {"reply": response, "tier": "groq_mixtral", "intent": intent}
    
    # Tier 3: Gemini Flash (Bypassed)
    response = get_gemini_response(system_prompt, user_message)
    if response:
        return {"reply": response, "tier": "gemini", "intent": intent}
        
    # Tier 3.5: Local LLM (qwen2.5:3b or other small local model)
    response = get_local_ollama_response(system_prompt, user_message, "qwen2.5:3b")
    if response:
        return {"reply": response, "tier": "local_ollama_qwen", "intent": intent}
        
    # Tier 4: Offline templates
    fallback = OFFLINE_TEMPLATES.get(intent, OFFLINE_TEMPLATES["general_fitness"])
    return {"reply": fallback, "tier": "offline", "intent": intent}
