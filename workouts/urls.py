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
]
