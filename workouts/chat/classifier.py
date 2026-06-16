import requests

INTENT_KEYWORDS = {
    "injury_pain": [
        "pain", "hurt", "injury", "sore", "ache", "pulled",
        "strain", "sprain", "swollen", "discomfort", "burning"
    ],
    "workout_plan": [
        "program", "routine", "workout plan", "training plan",
        "give me a plan", "weekly split", "push pull legs",
        "ppl", "bro split", "make me a routine"
    ],
    "nutrition_plan": [
        "meal plan", "what to eat", "diet plan", "food plan",
        "what should i eat", "eating plan", "nutrition plan"
    ],
    "biometrics": [
        "calories", "macros", "bmr", "tdee", "how much protein",
        "daily intake", "calorie target", "maintenance calories"
    ],
    "exercise_technique": [
        "how to do", "form", "technique", "tutorial",
        "proper way", "squat form", "deadlift", "bench press",
        "correct form", "biomechanics"
    ],
    "motivation": [
        "motivated", "giving up", "tired of", "burnout",
        "cant do it", "no energy", "feel like quitting",
        "whats the point", "depressed about gym", "lost motivation"
    ],
    "plateau": [
        "plateau", "stuck", "not losing", "same weight",
        "no progress", "weight not moving", "stalled"
    ],
    "supplement": [
        "creatine", "protein powder", "supplement", "pre workout",
        "whey", "bcaa", "mass gainer", "fat burner", "ashwagandha"
    ],
    "off_topic": [
        "politics", "movie", "cricket", "coding", "programming",
        "weather", "stock", "news", "relationship"
    ]
}

def keyword_classify(message: str) -> str | None:
    msg = message.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in msg)
        if score > 0:
            scores[intent] = score
    if not scores:
        return None
    return max(scores, key=scores.get)

def qwen_classify(message: str) -> str:
    VALID_INTENTS = list(INTENT_KEYWORDS.keys()) + ["general_fitness"]
    prompt = f"""You are a classifier for a fitness chatbot.
Classify the user message into exactly one of these labels:
{', '.join(VALID_INTENTS)}

Rules:
- Respond with ONLY the label
- No explanation, no punctuation, no extra words
- If unsure, respond: general_fitness

User message: {message}
Label:"""

    try:
        # qwen2.5:3b or similar <3b parameter local model
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
            timeout=4
        )
        result = r.json().get("response", "").strip().lower()
        # Clean up in case model output contains surrounding formatting
        for label in VALID_INTENTS:
            if label in result:
                return label
        return "general_fitness"
    except Exception:
        return "general_fitness"

def classify_intent(message: str) -> str:
    result = keyword_classify(message)
    if result:
        return result
    return qwen_classify(message)
