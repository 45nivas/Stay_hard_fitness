from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
import cv2
import mediapipe as mp
import numpy as np
from .rep_counter import RepCounter
from .models import UserProfile, ChatSession, ChatMessage
from .forms import UserProfileForm, ChatMessageForm
from .fitness_chatbot import FitnessChatbot
import json
import uuid

def home(request):
    if request.user.is_authenticated:
        return redirect('workout_selection')
    return redirect('login')

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

@login_required
def workout_selection(request):
    workouts = [
        {'name': 'Squats', 'slug': 'squats'},
        {'name': 'Push-ups', 'slug': 'pushups'},
        {'name': 'Bicep Curls', 'slug': 'bicep_curls'},
        {'name': 'Hammer Curls', 'slug': 'hammer_curls'},
        {'name': 'Side Raises', 'slug': 'side_raises'},
    ]
    return render(request, 'workout_selection.html', {
        'workouts': workouts
    })

@login_required
def workout_page(request, workout_name):
    return render(request, 'workout_page.html', {'workout_name': workout_name})

def gen_frames(workout_name):
    # Initialize MediaPipe and rep counter
    cap = cv2.VideoCapture(0)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize rep counter for specific workout
    counter = RepCounter(workout_name)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Process landmarks for rep counting
            if results.pose_landmarks:
                counter.process_frame(results.pose_landmarks)
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            # Add workout info overlay
            cv2.rectangle(image, (0, 0), (450, 140), (0, 0, 0), -1)
            cv2.putText(image, f'Workout: {workout_name.replace("_", " ").title()}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display reps based on workout type
            if workout_name in ["bicep_curls", "hammer_curls"]:
                cv2.putText(image, f'Left: {counter.left_rep_count} Right: {counter.right_rep_count}', 
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, f'Total Reps: {counter.rep_count}', 
                           (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.putText(image, f'Reps: {counter.rep_count}', 
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
            cv2.putText(image, f'Stage: {counter.stage if counter.stage else "Ready"}', 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add form tips
            tips = counter.get_feedback_text()
            for i, tip in enumerate(tips[:3]):  # Show only 3 tips
                cv2.putText(image, tip, (10, 150 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    except Exception as e:
        print(f"Error in video generation: {e}")
    finally:
        counter.cleanup()
        cap.release()

@login_required
def video_feed(request, workout_name):
    return StreamingHttpResponse(gen_frames(workout_name), content_type='multipart/x-mixed-replace; boundary=frame')

@login_required
def fitness_chat(request):
    """Fitness chatbot interface"""
    # Get or create chat session
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={'user': request.user}
    )
    
    # Initialize chatbot
    chatbot = FitnessChatbot()
    
    # Load existing user data from session
    if session.user_data:
        chatbot.user_data = session.user_data
    
    if request.method == 'POST':
        form = ChatMessageForm(request.POST)
        if form.is_valid():
            user_message = form.cleaned_data['message']
            
            # Process message with chatbot
            bot_response = chatbot.process_message(user_message)
            
            # Save chat message
            ChatMessage.objects.create(
                session=session,
                message=user_message,
                response=bot_response
            )
            
            # Update session user data
            session.user_data = chatbot.user_data
            session.save()
            
            # Return JSON response for AJAX
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'response': bot_response,
                    'user_data': chatbot.get_user_profile_summary()
                })
    else:
        form = ChatMessageForm()
    
    # Get chat history
    messages = session.messages.all()
    
    # Initialize welcome message if no messages
    welcome_message = None
    if not messages.exists():
        welcome_message = chatbot.initialize_conversation()
    
    context = {
        'form': form,
        'messages': messages,
        'welcome_message': welcome_message,
        'user_data': chatbot.get_user_profile_summary(),
        'session_id': session_id
    }
    
    return render(request, 'fitness_chat.html', context)


@login_required
def profile_setup(request):
    """Setup or update user profile"""
    try:
        profile = UserProfile.objects.get(user=request.user)
    except UserProfile.DoesNotExist:
        profile = None
    
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            profile = form.save(commit=False)
            profile.user = request.user
            profile.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('fitness_chat')
    else:
        form = UserProfileForm(instance=profile)
    
    return render(request, 'profile_setup.html', {'form': form, 'profile': profile})


@login_required
def clear_chat_session(request):
    """Clear current chat session"""
    session_id = request.session.get('chat_session_id')
    if session_id:
        try:
            session = ChatSession.objects.get(session_id=session_id, user=request.user)
            session.delete()
            del request.session['chat_session_id']
        except ChatSession.DoesNotExist:
            pass
    
    messages.success(request, 'Chat session cleared!')
    return redirect('fitness_chat')
