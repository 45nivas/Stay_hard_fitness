from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]
    
    FITNESS_LEVEL_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    
    GOAL_CHOICES = [
        ('weight_loss', 'Weight Loss'),
        ('muscle_gain', 'Muscle Gain'),
        ('strength', 'Build Strength'),
        ('endurance', 'Improve Endurance'),
        ('toning', 'Toning & Definition'),
        ('general_fitness', 'General Fitness'),
        ('bulking', 'Bulking'),
        ('cutting', 'Cutting'),
        ('maintaining', 'Maintaining'),
    ]
    
    WEAK_MUSCLE_CHOICES = [
        ('upper_chest', 'Upper Chest'),
        ('lower_chest', 'Lower Chest'),
        ('rear_delts', 'Rear Delts'),
        ('side_delts', 'Side Delts'),
        ('front_delts', 'Front Delts'),
        ('triceps', 'Triceps'),
        ('biceps', 'Biceps'),
        ('forearms', 'Forearms'),
        ('lats', 'Lats'),
        ('rhomboids', 'Rhomboids'),
        ('traps', 'Traps'),
        ('core', 'Core'),
        ('glutes', 'Glutes'),
        ('quadriceps', 'Quadriceps'),
        ('hamstrings', 'Hamstrings'),
        ('calves', 'Calves'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    age = models.IntegerField()
    height = models.FloatField(help_text="Height in cm")
    weight = models.FloatField(help_text="Weight in kg")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    fitness_level = models.CharField(max_length=20, choices=FITNESS_LEVEL_CHOICES)
    primary_goal = models.CharField(max_length=20, choices=GOAL_CHOICES)
    injuries_or_limitations = models.TextField(blank=True, help_text="Any injuries or physical limitations")
    available_time = models.IntegerField(help_text="Available workout time in minutes")
    weak_muscles = models.CharField(max_length=200, blank=True, help_text="Comma-separated list of weak muscle groups")
    equipment_available = models.TextField(blank=True, help_text="Available equipment (comma-separated)")
    calories_per_day = models.IntegerField(blank=True, null=True, help_text="Daily calorie intake")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    @property
    def bmi(self):
        """Calculate BMI"""
        height_m = self.height / 100
        return round(self.weight / (height_m * height_m), 1)
    
    @property
    def bmi_category(self):
        """Get BMI category"""
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

class WorkoutRecommendation(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    recommended_exercises = models.JSONField()
    difficulty_level = models.CharField(max_length=20)
    estimated_duration = models.IntegerField(help_text="Duration in minutes")
    focus_areas = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Workout for {self.user_profile.user.username}"


class ChatSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, unique=True)
    user_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chat session for {self.user.username}"


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"Message from {self.session.user.username} at {self.timestamp}"


# Calorie Tracking Models
class FoodItem(models.Model):
    name = models.CharField(max_length=200, unique=True)
    calories_per_100g = models.FloatField()
    protein = models.FloatField()
    carbs = models.FloatField()
    fat = models.FloatField()
    fiber = models.FloatField(default=0)
    sugar = models.FloatField(default=0)
    sodium = models.FloatField(default=0)  # in grams

    def __str__(self):
        return self.name


class MealLog(models.Model):
    MEAL_TYPE_CHOICES = [
        ('breakfast', 'Breakfast'),
        ('lunch', 'Lunch'),
        ('dinner', 'Dinner'),
        ('snack', 'Snack'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    food_item = models.ForeignKey(FoodItem, on_delete=models.CASCADE)
    quantity = models.FloatField()
    unit = models.CharField(max_length=32, blank=True, default="")
    meal_type = models.CharField(max_length=20, choices=MEAL_TYPE_CHOICES, default='snack')
    calories = models.FloatField(default=0)
    protein = models.FloatField(default=0)
    carbs = models.FloatField(default=0)
    fat = models.FloatField(default=0)
    fiber = models.FloatField(default=0)
    sugar = models.FloatField(default=0)
    sodium = models.FloatField(default=0)
    source = models.CharField(max_length=20, default='AI')  # 'AI' or 'USDA'
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.quantity} of {self.food_item.name} on {self.date} by {self.user.username}"


class DailySummary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField()
    total_calories = models.PositiveIntegerField(default=0)
    total_protein = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_carbohydrates = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_fats = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_fiber = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_sugar = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total_sodium = models.DecimalField(max_digits=10, decimal_places=3, default=0)

    class Meta:
        unique_together = ['user', 'date']

    def __str__(self):
        return f"Summary for {self.user.username} on {self.date}"
