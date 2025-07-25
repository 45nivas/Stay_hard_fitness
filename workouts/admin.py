from django.contrib import admin
from .models import UserProfile, ChatSession, ChatMessage, FoodItem, MealLog, DailySummary

# Register your models here.
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'age', 'gender', 'fitness_level', 'primary_goal']
    list_filter = ['gender', 'fitness_level', 'primary_goal']
    search_fields = ['user__username', 'user__email']

@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['user', 'session_id', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user__username', 'session_id']

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'timestamp', 'message_preview']
    list_filter = ['timestamp']
    search_fields = ['message', 'response']
    
    def message_preview(self, obj):
        return obj.message[:50] + "..." if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message Preview'

@admin.register(FoodItem)
class FoodItemAdmin(admin.ModelAdmin):
    list_display = ['name', 'calories_per_100g', 'protein', 'carbs', 'fat']
    search_fields = ['name']
    list_filter = ['calories_per_100g']

@admin.register(MealLog)
class MealLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'food_item', 'quantity', 'unit', 'calories', 'date']
    list_filter = ['date', 'food_item']
    search_fields = ['user__username', 'food_item__name']
    date_hierarchy = 'date'

@admin.register(DailySummary)
class DailySummaryAdmin(admin.ModelAdmin):
    list_display = ['user', 'date', 'total_calories', 'total_protein', 'total_carbohydrates', 'total_fats']
    list_filter = ['date']
    search_fields = ['user__username']
    date_hierarchy = 'date'
