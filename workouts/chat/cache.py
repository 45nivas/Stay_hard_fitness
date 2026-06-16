import hashlib
from django.core.cache import cache

STATIC_INTENTS = ["exercise_technique", "supplement", "off_topic"]

def get_cache_key(intent: str, message: str) -> str:
    content = f"{intent}:{message.lower().strip()}"
    return "oschat_" + hashlib.md5(content.encode()).hexdigest()

def get_cached_response(intent: str, message: str):
    return cache.get(get_cache_key(intent, message))

def set_cached_response(intent: str, message: str, response: str):
    ttl = 86400 if intent in STATIC_INTENTS else 3600
    cache.set(get_cache_key(intent, message), response, ttl)
