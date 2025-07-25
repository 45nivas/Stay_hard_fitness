from django import forms
from .models import UserProfile

class UserProfileForm(forms.ModelForm):
    weak_muscles = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., upper_chest, rear_delts, core',
            'help_text': 'Comma-separated list of muscle groups you want to focus on'
        }),
        help_text='Enter muscle groups you want to target (comma-separated)'
    )
    
    equipment_available = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., dumbbells, resistance_bands, yoga_mat',
            'help_text': 'List your available equipment'
        }),
        help_text='Enter available equipment (comma-separated)'
    )
    
    class Meta:
        model = UserProfile
        fields = ['age', 'height', 'weight', 'gender', 'fitness_level', 
                 'primary_goal', 'injuries_or_limitations', 'available_time',
                 'weak_muscles', 'equipment_available', 'calories_per_day']
        widgets = {
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': 13, 'max': 100}),
            'height': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': 120, 'max': 250}),
            'weight': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'min': 30, 'max': 300}),
            'gender': forms.Select(attrs={'class': 'form-control'}),
            'fitness_level': forms.Select(attrs={'class': 'form-control'}),
            'primary_goal': forms.Select(attrs={'class': 'form-control'}),
            'injuries_or_limitations': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'available_time': forms.NumberInput(attrs={'class': 'form-control', 'min': 15, 'max': 180}),
            'calories_per_day': forms.NumberInput(attrs={'class': 'form-control', 'min': 1200, 'max': 5000, 'required': False}),
        }
        labels = {
            'age': 'Age (years)',
            'height': 'Height (cm)',
            'weight': 'Weight (kg)',
            'gender': 'Gender',
            'fitness_level': 'Current Fitness Level',
            'primary_goal': 'Primary Fitness Goal',
            'injuries_or_limitations': 'Injuries or Physical Limitations (optional)',
            'available_time': 'Available Workout Time (minutes)',
        }


class ChatMessageForm(forms.Form):
    message = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control chat-input',
            'placeholder': 'Ask me anything about fitness, workouts, or nutrition...',
            'autocomplete': 'off',
            'maxlength': '500'
        }),
        max_length=500,
        required=True
    )
