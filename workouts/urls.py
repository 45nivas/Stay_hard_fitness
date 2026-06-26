from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('signup/', views.signup, name='signup'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('workouts/', views.workout_selection, name='workout_selection'),
    path('workout/<str:workout_name>/', views.workout_page, name='workout_page'),
    path('video_feed/<str:workout_name>/', views.video_feed, name='video_feed'),
    # Profile and Chat URLs
    path('profile/', views.profile_setup, name='profile_setup'),
    path('fitness-chat/', views.fitness_chat, name='fitness_chat'),
    path('clear-chat/', views.clear_chat_session, name='clear_chat_session'),
    # One Rep Max Calculator
    path('one-rep-max/', views.one_rep_max_calculator, name='one_rep_max'),
    # Carb Cycling Calculator
    path('carb-cycling/', views.carb_cycling_calculator, name='carb_cycling'),
    # Calorie Tracker URLs
    path('calorie-tracker/', views.calorie_tracker, name='calorie_tracker'),
    path('api/voice-log/', views.log_meal_from_voice, name='voice_log'),
    path('api/confirm-meal/', views.confirm_meal, name='confirm_meal'),
    path('api/get-daily-summary/', views.get_daily_summary, name='get_daily_summary'),
    path('api/get-daily-meals/', views.get_daily_meals, name='get_daily_meals'),
    path('api/delete-meal/', views.delete_meal, name='delete_meal'),
    path('api/recalculate-meals/', views.recalculate_meals, name='recalculate_meals'),
    # Pose Correction URLs
    path('pose-correction/', views.pose_correction, name='pose_correction'),
    path('posture-analysis/', views.posture_analysis, name='posture_analysis'),
    path('api/workout-stats/', views.workout_stats_api, name='workout_stats_api'),
    path('api/save-posture-analysis/', views.save_posture_analysis, name='save_posture_analysis'),
    path('analytics/', views.analytics_dashboard, name='analytics'),
    path('api/generate-recommendation/', views.generate_adaptive_workout, name='generate_recommendation'),
    path('body-analysis/', views.body_analysis_view, name='body_analysis'),
    path('api/analyse-body/', views.analyse_body_api, name='analyse_body_api'),
    path('api/voice-log-workout/', views.voice_log_workout_api, name='voice_log_workout_api'),
    path('api/parse-workout-voice/', views.parse_workout_voice_api, name='parse_workout_voice_api'),
    path('api/confirm-workout-log/', views.confirm_workout_log_api, name='confirm_workout_log_api'),
    path('api/transcribe-audio/', views.transcribe_audio, name='transcribe_audio'),
    # Decoupled APIs
    path('api/login/', views.api_login, name='api_login'),
    path('api/signup/', views.api_signup, name='api_signup'),
    path('api/logout/', views.api_logout, name='api_logout'),
    path('api/user-status/', views.api_user_status, name='api_user_status'),
    path('api/workouts/', views.api_workout_selection, name='api_workout_selection'),
    path('api/profile-setup/', views.api_profile_get_or_post, name='api_profile_get_or_post'),
    path('api/chat/', views.api_fitness_chat, name='api_fitness_chat'),
    path('api/one-rep-max/', views.api_one_rep_max_calculator, name='api_one_rep_max_calculator'),
    path('api/carb-cycling/', views.api_carb_cycling_calculator, name='api_carb_cycling_calculator'),
    path('api/get-workouts/', views.api_get_workout_logs, name='api_get_workout_logs'),
    path('api/delete-workout/', views.api_delete_workout_log, name='api_delete_workout_log'),
    path('api/exercise-progress/<str:exercise_name>/', views.exercise_progress_api, name='exercise_progress_api'),
]
