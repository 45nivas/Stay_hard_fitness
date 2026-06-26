import json
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.utils import timezone
from unittest.mock import patch, MagicMock

from workouts.models import (
    UserProfile, ChatSession, ChatMessage, FoodItem, 
    MealLog, DailySummary, PostureAnalysis, WorkoutLog, 
    WorkoutRecommendation, FoodPreference
)
from workouts.forms import UserProfileForm

class StayHardFitnessTestSuite(TestCase):
    def setUp(self):
        self.client = Client()
        self.username = 'testathlete'
        self.password = 'StrongPass123!'
        self.user = User.objects.create_user(username=self.username, password=self.password)
        
        # Create user profile
        self.profile = UserProfile.objects.create(
            user=self.user,
            age=28,
            height=180.0,
            weight=82.5,
            gender='M',
            fitness_level='intermediate',
            primary_goal='muscle_gain',
            available_time=60,
            weak_muscles='Biceps,Legs',
            equipment_available='dumbbells,barbell'
        )

    def test_home_redirects(self):
        # Unauthenticated redirects to login
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)
        
        # Authenticated redirects to workout selection
        self.client.login(username=self.username, password=self.password)
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse('workout_selection'))

    def test_signup_view_get(self):
        response = self.client.get(reverse('signup'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'signup.html')

    def test_signup_view_post(self):
        # Successful signup redirects to login (302)
        valid_post = {
            'username': 'uniqueathlete',
            'password1': 'TestingStrongPass123!',
            'password2': 'TestingStrongPass123!',
        }
        response = self.client.post(reverse('signup'), valid_post)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username='uniqueathlete').exists())

    def test_login_view_get(self):
        response = self.client.get(reverse('login'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'login.html')

    def test_workout_selection_view(self):
        # Requires login
        response = self.client.get(reverse('workout_selection'))
        self.assertEqual(response.status_code, 302)
        
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(reverse('workout_selection'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'workout_selection.html')
        
        # Dashboard displays the module cards
        self.assertContains(response, 'AI Coach')
        self.assertContains(response, 'Voice Macros')
        self.assertContains(response, 'Pose Correction')

    def test_workout_page_view(self):
        self.client.login(username=self.username, password=self.password)
        for exercise in ['squats', 'pushups', 'bicep_curls']:
            response = self.client.get(reverse('workout_page', kwargs={'workout_name': exercise}))
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'workout_page.html')

    def test_video_feed_stream(self):
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(reverse('video_feed', kwargs={'workout_name': 'squats'}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'multipart/x-mixed-replace; boundary=frame')

    def test_profile_setup_view(self):
        self.client.login(username=self.username, password=self.password)
        
        # GET returns form
        response = self.client.get(reverse('profile_setup'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'profile_setup.html')
        
        # POST updates profile
        update_data = {
            'age': 29,
            'height': 181.5,
            'weight': 84.0,
            'gender': 'M',
            'fitness_level': 'advanced',
            'primary_goal': 'muscle_gain',
            'available_time': 90,
            'weak_muscles': 'Triceps,Rhomboids',
            'equipment_available': 'dumbbells,barbell,cables',
            'calories_per_day': 3000
        }
        response = self.client.post(reverse('profile_setup'), update_data)
        self.assertEqual(response.status_code, 302) # Redirects to fitness_chat
        
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.age, 29)
        self.assertEqual(self.profile.height, 181.5)
        self.assertEqual(self.profile.weight, 84.0)
        self.assertEqual(self.profile.fitness_level, 'advanced')

    @patch('workouts.fitness_chatbot.FitnessChatbot._query_gemini_1_5')
    def test_fitness_chat_service(self, mock_gemini):
        mock_gemini.return_value = "Mocked trainer response focusing on core training."
        self.client.login(username=self.username, password=self.password)
        
        # Get Chat page
        response = self.client.get(reverse('fitness_chat'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fitness_chat.html')
        
        # Post a fitness message via AJAX
        chat_data = {'message': 'What is a good squat progression?'}
        response = self.client.post(
            reverse('fitness_chat'), 
            chat_data, 
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertTrue(resp_json['success'])
        self.assertIn('response', resp_json)
        
        # Verify messages are saved
        chat_session = ChatSession.objects.get(user=self.user)
        self.assertTrue(ChatMessage.objects.filter(session=chat_session).exists())
        
        # Test clear chat session
        clear_response = self.client.get(reverse('clear_chat_session'))
        self.assertEqual(clear_response.status_code, 302)
        self.assertFalse(ChatMessage.objects.filter(session=chat_session).exists())

    def test_one_rep_max_calculator_get(self):
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(reverse('one_rep_max'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'one_rep_max.html')

    def test_carb_cycling_calculator_post(self):
        self.client.login(username=self.username, password=self.password)
        calc_data = {
            'age': 25,
            'gender': 'male',
            'height': 175,
            'weight': 75,
            'height_unit': 'cm',
            'weight_unit': 'kg',
            'activity_level': 'moderate',
            'goal': 'muscle_gain',
            'training_days': 4
        }
        response = self.client.post(reverse('carb_cycling'), calc_data)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'carb_cycling.html')
        self.assertContains(response, 'HIGH CARB')
        self.assertContains(response, 'Fixed Protein')
        self.assertContains(response, 'Fixed Fats')

    def test_calorie_tracker_view(self):
        self.client.login(username=self.username, password=self.password)
        response = self.client.get(reverse('calorie_tracker'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'calorie_tracker.html')

    @patch('workouts.views.calories.query_gemini_for_foods')
    def test_log_meal_from_voice_and_rec_apis(self, mock_gemini_foods):
        # 1. Mock Gemini returningParsed Food list
        mock_gemini_foods.return_value = [
            {
                "food": "chicken breast",
                "quantity": 200,
                "unit": "g",
                "calories": 330,
                "protein": 62,
                "carbs": 0,
                "fat": 7.2,
                "fiber": 0,
                "sugar": 0,
                "sodium": 0.148,
                "source": "AI"
            }
        ]
        
        self.client.login(username=self.username, password=self.password)
        
        # Test logging a meal
        voice_payload = {"text": "I ate 200g chicken breast"}
        response = self.client.post(
            reverse('voice_log'),
            data=json.dumps(voice_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        resp_data = response.json()
        self.assertEqual(resp_data['mode'], 'picker')
        self.assertEqual(len(resp_data['candidates']), 3)

        # Confirm the choice so it logs the meal
        confirm_payload = {
            "food_query": "chicken",
            "chosen_food_name": "chicken breast",
            "chosen_food_data": {
                "calories": 330,
                "protein": 62,
                "carbs": 0,
                "fat": 7.2,
                "fiber": 0,
                "sodium": 0.148
            },
            "quantity": 200,
            "unit": "g"
        }
        response = self.client.post(
            reverse('confirm_meal'),
            data=json.dumps(confirm_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        resp_data = response.json()
        self.assertEqual(len(resp_data['logged']), 1)
        self.assertEqual(resp_data['logged'][0]['food'], 'chicken breast')
        
        # Verify db log
        self.assertTrue(MealLog.objects.filter(user=self.user, food_item__name='chicken breast').exists())
        meal_id = MealLog.objects.get(user=self.user, food_item__name='chicken breast').id

        # Test daily summary API
        response = self.client.get(reverse('get_daily_summary'))
        self.assertEqual(response.status_code, 200)
        summary = response.json()
        self.assertEqual(summary['total_calories'], 330)
        self.assertEqual(summary['total_protein'], 62.0)

        # Test daily meals list API
        response = self.client.get(reverse('get_daily_meals'))
        self.assertEqual(response.status_code, 200)
        meals_list = response.json()
        self.assertEqual(len(meals_list['meals']), 1)
        self.assertEqual(meals_list['meals'][0]['food'], 'chicken breast')

        # Test recalculate API
        response = self.client.post(reverse('recalculate_meals'))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])

        # Test delete meal API
        response = self.client.post(
            reverse('delete_meal'),
            data=json.dumps({"id": meal_id}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        self.assertFalse(MealLog.objects.filter(id=meal_id).exists())

    def test_save_posture_analysis_and_workout_stats(self):
        self.client.login(username=self.username, password=self.password)
        
        # Post simulated workout completion
        posture_payload = {
            'exercise_name': 'squats',
            'rep_count': 10,
            'stage': 'Complete',
            'feedback': ['Good knee stability', 'Proper stance']
        }
        
        response = self.client.post(
            reverse('save_posture_analysis'),
            data=json.dumps(posture_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'success')
        
        # Check databases
        self.assertTrue(PostureAnalysis.objects.filter(user=self.user, exercise_name='Squats').exists())
        self.assertTrue(WorkoutLog.objects.filter(user=self.user, exercise_name='Squat', muscle_group='Legs').exists())
        
        # Set simulated active session in WORKOUT_STATS
        from workouts.views import WORKOUT_STATS
        WORKOUT_STATS[self.user.id] = {
            'workout_name': 'squats',
            'rep_count': 10,
            'left_rep_count': 0,
            'right_rep_count': 0,
            'stage': 'Complete',
            'feedback': ['Good knee stability'],
            'active': True
        }

        # Test stats API
        response = self.client.get(reverse('workout_stats_api'))
        self.assertEqual(response.status_code, 200)
        stats = response.json()
        self.assertEqual(stats['workout_name'], 'squats')
        
        # Test pose correction page
        response = self.client.get(reverse('pose_correction'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'pose_correction.html')

        # Test posture analysis list page
        response = self.client.get(reverse('posture_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'posture_analysis.html')

    def test_analytics_dashboard_view(self):
        self.client.login(username=self.username, password=self.password)
        
        # Populate workout log history to test chart statistics
        WorkoutLog.objects.create(
            user=self.user,
            exercise_name='Squats',
            sets=4,
            reps=10,
            weight=100.0,
            muscle_group='Legs'
        )
        WorkoutLog.objects.create(
            user=self.user,
            exercise_name='Bicep Curls',
            sets=3,
            reps=12,
            weight=15.0,
            muscle_group='Biceps'
        )
        
        response = self.client.get(reverse('analytics'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'analytics.html')
        self.assertContains(response, 'Legs')
        self.assertContains(response, 'Biceps')

    def test_generate_adaptive_workout_api(self):
        self.client.login(username=self.username, password=self.password)
        
        # Verify offline fallback recommendations
        response = self.client.get(reverse('generate_recommendation'))
        self.assertEqual(response.status_code, 200)
        resp_data = response.json()
        self.assertEqual(resp_data['status'], 'success')
        self.assertIn('routine', resp_data)
        self.assertIn('difficulty', resp_data)
        
        # Check that WorkoutRecommendation object is persisted
        self.assertTrue(WorkoutRecommendation.objects.filter(user_profile=self.profile).exists())

    @patch('workouts.views.calories.query_gemini_for_foods')
    def test_adaptive_food_preference_system(self, mock_gemini_foods):
        # Setup mock for query_gemini_for_foods
        mock_gemini_foods.return_value = [
            {
                "food": "rolled oats",
                "quantity": 200,
                "unit": "g",
                "calories": 712,
                "protein": 24,
                "carbs": 120,
                "fat": 12,
                "fiber": 20,
                "sodium": 0.004
            }
        ]
        
        self.client.login(username=self.username, password=self.password)
        
        # 1. No FoodPreference exists yet: should return mode='picker'
        voice_payload = {"text": "I ate 200g rolled oats"}
        response = self.client.post(
            reverse('voice_log'),
            data=json.dumps(voice_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data['mode'], 'picker')
        self.assertEqual(res_data['food_query'], 'oats')  # 'rolled oats' normalized to 'oats'
        self.assertEqual(len(res_data['candidates']), 3)
        
        # 2. Confirm candidate choice to create preference (log_count = 1)
        confirm_payload = {
            "food_query": "oats",
            "chosen_food_name": "Saffola Oats",
            "chosen_food_data": {
                "calories": 712,
                "protein": 24,
                "carbs": 120,
                "fat": 12,
                "fiber": 20,
                "sodium": 0.004
            },
            "quantity": 200,
            "unit": "g"
        }
        
        response = self.client.post(
            reverse('confirm_meal'),
            data=json.dumps(confirm_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        
        # Verify db records
        pref = FoodPreference.objects.get(user=self.user, food_query='oats')
        self.assertEqual(pref.log_count, 1)
        self.assertEqual(pref.preferred_food_name, 'Saffola Oats')
        # Base calories per 100g = 712 / 2.0 = 356.0
        self.assertEqual(pref.preferred_food_data['calories'], 356.0)
        
        # Verify DailySummary has been populated
        self.assertTrue(DailySummary.objects.filter(user=self.user).exists())

        # 3. Confirm 2 more times to bring log_count to 3
        for _ in range(2):
            self.client.post(
                reverse('confirm_meal'),
                data=json.dumps(confirm_payload),
                content_type='application/json'
            )
            
        pref.refresh_from_db()
        self.assertEqual(pref.log_count, 3)

        # 4. Request voice-log again: should skip fallback chain and return auto mode!
        # Let's say user requests a different quantity: 100g oats
        mock_gemini_foods.return_value = [
            {
                "food": "rolled oats",
                "quantity": 100,
                "unit": "g"
            }
        ]
        
        response = self.client.post(
            reverse('voice_log'),
            data=json.dumps(voice_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data['mode'], 'auto')
        self.assertEqual(res_data['logged'][0]['food'], 'Saffola Oats')
        # Scaled to 100g: 356.0 calories
        self.assertEqual(res_data['logged'][0]['calories'], 356.0)
        
        # Verify log count incremented to 4
        pref.refresh_from_db()
        self.assertEqual(pref.log_count, 4)


class BodyVisionTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.username = 'visionathlete'
        self.password = 'StrongVisionPass123!'
        self.user = User.objects.create_user(username=self.username, password=self.password)
        self.profile = UserProfile.objects.create(
            user=self.user,
            age=25,
            height=175.0,
            weight=70.0,
            gender='M',
            fitness_level='intermediate',
            primary_goal='muscle_gain',
            available_time=60,
            weak_muscles='Chest'
        )

    def test_body_analysis_view_requires_login(self):
        response = self.client.get(reverse('body_analysis'))
        self.assertEqual(response.status_code, 302)

        self.client.login(username=self.username, password=self.password)
        response = self.client.get(reverse('body_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'body_analysis.html')

    def test_analyse_body_api_requires_login_and_post(self):
        # Unauthenticated POST redirects to login
        response = self.client.post(reverse('analyse_body_api'))
        self.assertEqual(response.status_code, 302)

        self.client.login(username=self.username, password=self.password)
        
        # GET returns 405 Method Not Allowed
        response = self.client.get(reverse('analyse_body_api'))
        self.assertEqual(response.status_code, 405)

    def test_analyse_body_api_missing_photo(self):
        self.client.login(username=self.username, password=self.password)
        response = self.client.post(reverse('analyse_body_api'), {})
        self.assertEqual(response.status_code, 400)
        self.assertIn('No photo uploaded', response.json()['error'])

    def test_analyse_body_api_invalid_file_type(self):
        self.client.login(username=self.username, password=self.password)
        from django.core.files.uploadedfile import SimpleUploadedFile
        photo = SimpleUploadedFile("physique.txt", b"not an image", content_type="text/plain")
        response = self.client.post(reverse('analyse_body_api'), {'photo': photo})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Upload JPG, PNG or WEBP only', response.json()['error'])

    def test_analyse_body_api_file_too_large(self):
        self.client.login(username=self.username, password=self.password)
        from django.core.files.uploadedfile import SimpleUploadedFile
        large_bytes = b"0" * (10 * 1024 * 1024 + 1)
        photo = SimpleUploadedFile("physique.jpg", large_bytes, content_type="image/jpeg")
        response = self.client.post(reverse('analyse_body_api'), {'photo': photo})
        self.assertEqual(response.status_code, 400)
        self.assertIn('File too large', response.json()['error'])

    @patch('workouts.views.posture.extract_landmarks_and_symmetry')
    @patch('workouts.views.posture.analyse_with_gemini_vision')
    def test_analyse_body_api_success_with_gemini(self, mock_gemini, mock_landmarks):
        self.client.login(username=self.username, password=self.password)
        from django.core.files.uploadedfile import SimpleUploadedFile
        photo = SimpleUploadedFile("physique.jpg", b"valid_bytes", content_type="image/jpeg")
        
        mock_landmarks.return_value = {
            "shoulder_width_px": 200.0,
            "hip_width_px": 120.0,
            "taper_ratio": 1.667,
            "arm_symmetry_score": 95.0,
            "leg_symmetry_score": 93.0,
            "torso_height_px": 400.0,
            "landmarks_detected": True
        }
        
        gemini_response = {
            "muscle_scores": {
                "chest": 5, "shoulders": 8, "biceps": 6, "triceps": 7,
                "back_width": 7, "back_thickness": 6, "core": 7,
                "quads": 4, "hamstrings": 3, "calves": 3
            },
            "weak_groups": ["quads", "hamstrings", "calves"],
            "dominant_groups": ["shoulders", "triceps", "back_width"],
            "body_type": "mesomorph",
            "taper_assessment": "V-taper",
            "priority_recommendation": "Focus on lagging posterior chain and lower body development.",
            "suggested_split": "Push Pull Legs with lower body focus",
            "confidence": "high",
            "disclaimer": "AI visual estimation only."
        }
        mock_gemini.return_value = gemini_response

        response = self.client.post(reverse('analyse_body_api'), {'photo': photo})
        self.assertEqual(response.status_code, 200)
        
        res_json = response.json()
        self.assertTrue(res_json['success'])
        self.assertEqual(res_json['analysis']['body_type'], 'mesomorph')
        self.assertEqual(res_json['analysis']['landmark_data']['taper_ratio'], 1.667)
        
        # Verify user profile weak_muscles got updated
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.weak_muscles, "quads, hamstrings, calves")

    @patch('workouts.views.posture.extract_landmarks_and_symmetry')
    @patch('workouts.views.posture.analyse_with_gemini_vision')
    def test_analyse_body_api_fallback_when_gemini_fails(self, mock_gemini, mock_landmarks):
        self.client.login(username=self.username, password=self.password)
        from django.core.files.uploadedfile import SimpleUploadedFile
        photo = SimpleUploadedFile("physique.jpg", b"valid_bytes", content_type="image/jpeg")
        
        mock_landmarks.return_value = {
            "shoulder_width_px": 200.0,
            "hip_width_px": 120.0,
            "taper_ratio": 1.667,
            "arm_symmetry_score": 95.0,
            "leg_symmetry_score": 93.0,
            "torso_height_px": 400.0,
            "landmarks_detected": True
        }
        
        # Gemini fails by returning None
        mock_gemini.return_value = None

        response = self.client.post(reverse('analyse_body_api'), {'photo': photo})
        self.assertEqual(response.status_code, 200)
        
        res_json = response.json()
        self.assertTrue(res_json['success'])
        self.assertEqual(res_json['analysis']['confidence'], 'low')  # Low confidence indicates fallback
        self.assertEqual(res_json['analysis']['taper_assessment'], 'V-taper')
        self.assertEqual(res_json['analysis']['weak_groups'], ["calves", "hamstrings"])
        
        # Verify user profile weak_muscles got updated with fallback weak groups
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.weak_muscles, "calves, hamstrings")

    @patch('workouts.views.posture.extract_landmarks_and_symmetry')
    def test_analyse_body_api_fails_when_landmarks_fail(self, mock_landmarks):
        self.client.login(username=self.username, password=self.password)
        from django.core.files.uploadedfile import SimpleUploadedFile
        photo = SimpleUploadedFile("physique.jpg", b"invalid_bytes", content_type="image/jpeg")
        
        # Landmarks extraction returns None
        mock_landmarks.return_value = None

        response = self.client.post(reverse('analyse_body_api'), {'photo': photo})
        self.assertEqual(response.status_code, 422)
        self.assertIn('Could not detect a body', response.json()['error'])


class VoiceWorkoutLoggerTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.username = 'voiceathlete'
        self.password = 'StrongVoicePass123!'
        self.user = User.objects.create_user(username=self.username, password=self.password)

    def test_voice_log_workout_api_requires_login_and_post(self):
        # Unauthenticated redirects to login
        response = self.client.post(reverse('voice_log_workout_api'))
        self.assertEqual(response.status_code, 302)

        self.client.login(username=self.username, password=self.password)
        
        # GET returns 405 Method Not Allowed
        response = self.client.get(reverse('voice_log_workout_api'))
        self.assertEqual(response.status_code, 405)

    def test_voice_log_workout_api_missing_text(self):
        self.client.login(username=self.username, password=self.password)
        
        # Empty text
        response = self.client.post(
            reverse('voice_log_workout_api'),
            data=json.dumps({"text": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("No workout transcript text provided", response.json()["error"])

    @patch('workouts.views.workouts.parse_workout_transcript_local_llm')
    def test_voice_log_workout_api_success_with_llm(self, mock_llm):
        self.client.login(username=self.username, password=self.password)
        
        mock_llm.return_value = [
            {
                "exercise_name": "Incline Bench Press",
                "sets": 3,
                "reps": 12,
                "weight_value": 100.0,
                "weight_unit": "lbs",
                "muscle_group": "Chest"
            },
            {
                "exercise_name": "Squats",
                "sets": 4,
                "reps": 8,
                "weight_value": 80.0,
                "weight_unit": "kg",
                "muscle_group": "Legs"
            }
        ]

        response = self.client.post(
            reverse('voice_log_workout_api'),
            data=json.dumps({"text": "random gym yapping text"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertTrue(res_data['success'])
        self.assertEqual(res_data['source'], 'local_llm')
        self.assertEqual(len(res_data['logged']), 2)
        
        self.assertEqual(res_data['logged'][0]['exercise_name'], 'Incline Bench Press')
        self.assertEqual(res_data['logged'][0]['weight'], 45.4)
        
        self.assertEqual(res_data['logged'][1]['exercise_name'], 'Squat')
        self.assertEqual(res_data['logged'][1]['weight'], 80.0)

        logs = WorkoutLog.objects.filter(user=self.user)
        self.assertEqual(logs.count(), 2)

    @patch('workouts.views.workouts.parse_workout_transcript_local_llm')
    def test_voice_log_workout_api_fallback_regex(self, mock_llm):
        self.client.login(username=self.username, password=self.password)
        
        mock_llm.return_value = None
        yapping_text = "Today I went to the gym. Warmed up, and did 3 sets of 12 reps of bench press at 100 lbs. Then took rest. Later did squats 4x8 with 80 kg."
        
        response = self.client.post(
            reverse('voice_log_workout_api'),
            data=json.dumps({"text": yapping_text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertTrue(res_data['success'])
        self.assertEqual(res_data['source'], 'regex_fallback')
        self.assertEqual(len(res_data['logged']), 2)
        
        self.assertEqual(res_data['logged'][0]['exercise_name'], 'Bench Press')
        self.assertEqual(res_data['logged'][0]['weight'], 45.4)
        self.assertEqual(res_data['logged'][0]['muscle_group'], 'Chest')
        
        self.assertEqual(res_data['logged'][1]['exercise_name'], 'Squat')
        self.assertEqual(res_data['logged'][1]['weight'], 80.0)
        self.assertEqual(res_data['logged'][1]['muscle_group'], 'Legs')

    @patch('workouts.views.workouts.parse_workout_transcript_local_llm')
    def test_voice_log_workout_api_invalid_text(self, mock_llm):
        self.client.login(username=self.username, password=self.password)
        
        mock_llm.return_value = None
        invalid_text = "I drank a water bottle and talked to the trainer today."

        response = self.client.post(
            reverse('voice_log_workout_api'),
            data=json.dumps({"text": invalid_text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("No exercises could be parsed", response.json()["error"])

    @patch('workouts.views.workouts.parse_workout_transcript_local_llm')
    def test_parse_workout_voice_api_success(self, mock_llm):
        self.client.login(username=self.username, password=self.password)
        mock_llm.return_value = None
        
        # Voice input containing numberless exercises and a bodyweight exercise
        text = "I did incline bench press and 3 sets of 10 pull ups and then 10 push ups"
        
        response = self.client.post(
            reverse('parse_workout_voice_api'),
            data=json.dumps({"text": text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertTrue(res_data['success'])
        
        parsed = res_data['parsed']
        self.assertEqual(len(parsed), 3)
        
        # 1. Incline Bench Press should default to 3 sets of 10 reps @ 0.0kg
        self.assertEqual(parsed[0]['exercise_name'], 'Incline Bench Press')
        self.assertEqual(len(parsed[0]['sets']), 3)
        self.assertEqual(parsed[0]['sets'][0]['reps'], 10)
        self.assertEqual(parsed[0]['sets'][0]['weight_value'], 0.0)
        self.assertEqual(parsed[0]['muscle_group'], 'Chest')
        
        # 2. Pull Ups -> Pull Up
        self.assertEqual(parsed[1]['exercise_name'], 'Pull Up')
        self.assertEqual(len(parsed[1]['sets']), 3)
        self.assertEqual(parsed[1]['sets'][0]['reps'], 10)
        
        # 3. Push Ups -> Push Up
        self.assertEqual(parsed[2]['exercise_name'], 'Push Up')
        self.assertEqual(len(parsed[2]['sets']), 3)
        self.assertIn(parsed[2]['sets'][0]['reps'], [0, 10])

        # Database should still be empty for today's logs
        self.assertEqual(WorkoutLog.objects.filter(user=self.user).count(), 0)

    def test_confirm_workout_log_api_success(self):
        self.client.login(username=self.username, password=self.password)
        
        payload = {
            "exercises": [
                {
                    "exercise_name": "Incline Bench Press",
                    "sets": 3,
                    "reps": 10,
                    "weight_value": 60.0,
                    "weight_unit": "kg",
                    "muscle_group": "Chest"
                },
                {
                    "exercise_name": "Pull Ups",
                    "sets": 3,
                    "reps": 12,
                    "weight_value": 0.0,
                    "weight_unit": "kg",
                    "muscle_group": "Back"
                }
            ]
        }
        
        response = self.client.post(
            reverse('confirm_workout_log_api'),
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertTrue(res_data['success'])
        self.assertEqual(len(res_data['logged']), 2)
        
        # DB check
        logs = WorkoutLog.objects.filter(user=self.user).order_by('id')
        self.assertEqual(logs.count(), 2)
        self.assertEqual(logs[0].exercise_name, "Incline Bench Press")
        self.assertEqual(logs[0].sets, 3)
        self.assertEqual(logs[0].reps, 10)
        self.assertEqual(logs[0].weight, 60.0)
        self.assertEqual(logs[1].exercise_name, "Pull Up")
        self.assertEqual(logs[1].reps, 12)




