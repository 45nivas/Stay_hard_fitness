import math
import threading
import time
from queue import Queue
import numpy as np

# Optional dependencies for voice feedback
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Text-to-speech not available - voice feedback disabled")

class WorkoutCoach:
    def __init__(self):
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.8)
                self.voice_queue = Queue()
                self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
                self.voice_thread.start()
                self.tts_enabled = True
            except Exception as e:
                print(f"TTS initialization failed: {e}")
                self.tts_enabled = False
        else:
            self.tts_enabled = False
        
    def _voice_worker(self):
        if not self.tts_enabled:
            return
        while True:
            message = self.voice_queue.get()
            if message is None:
                break
            if self.tts_enabled:
                try:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
            self.voice_queue.task_done()
            
    def speak(self, message):
        """Add voice feedback to queue (if TTS is available)"""
        if self.tts_enabled:
            try:
                self.voice_queue.put(message)
            except Exception:
                pass  # Silently fail if voice is not available
        
    def stop(self):
        """Stop voice feedback"""
        if self.tts_enabled:
            try:
                self.voice_queue.put(None)
            except Exception:
                pass

class RepCounter:
    def __init__(self, workout_type):
        self.workout_type = workout_type
        self.rep_count = 0
        self.left_rep_count = 0
        self.right_rep_count = 0
        self.stage = None
        self.left_stage = None
        self.right_stage = None
        self.coach = WorkoutCoach()
        self.last_feedback_time = 0
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points using numpy like in Flask app"""
        a = np.array(a)
        b = np.array(b) 
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle
    
    def count_squats(self, landmarks):
        try:
            # Get coordinates exactly like Flask app
            hip = [landmarks[23].x, landmarks[23].y]  # LEFT_HIP
            knee = [landmarks[25].x, landmarks[25].y]  # LEFT_KNEE
            ankle = [landmarks[27].x, landmarks[27].y]  # LEFT_ANKLE
            
            angle = self.calculate_angle(hip, knee, ankle)
            
            # Exact logic from Flask app
            if angle > 160:
                self.stage = "down"
            if angle < 100 and self.stage == 'down':
                self.stage = "up"
                self.rep_count += 1
                self.coach.speak(f"Great squat! {self.rep_count} reps")
                
            # Form feedback
            current_time = time.time()
            if current_time - self.last_feedback_time > 5:
                if angle < 80:
                    self.coach.speak("Perfect depth! Keep it up")
                    self.last_feedback_time = current_time
                elif angle > 120 and self.stage == "down":
                    self.coach.speak("Go deeper for better results")
                    self.last_feedback_time = current_time
                    
        except Exception as e:
            pass
            
    def count_pushups(self, landmarks):
        try:
            # Get coordinates exactly like Flask app
            shoulder = [landmarks[11].x, landmarks[11].y]  # LEFT_SHOULDER
            elbow = [landmarks[13].x, landmarks[13].y]     # LEFT_ELBOW
            wrist = [landmarks[15].x, landmarks[15].y]     # LEFT_WRIST
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Exact logic from Flask app
            if angle > 160:
                self.stage = "down"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.rep_count += 1
                self.coach.speak(f"Excellent push-up! {self.rep_count} reps")
                
            # Form feedback
            current_time = time.time()
            if current_time - self.last_feedback_time > 4:
                if angle < 20:
                    self.coach.speak("Perfect form! Full range of motion")
                    self.last_feedback_time = current_time
                elif 60 < angle < 120:
                    self.coach.speak("Go lower for better results")
                    self.last_feedback_time = current_time
                    
        except Exception as e:
            pass
            
    def count_bicep_curls(self, landmarks):
        try:
            # Get coordinates for both arms like Flask app (mirrored)
            left_shoulder = [landmarks[12].x, landmarks[12].y]  # RIGHT_SHOULDER (mirrored)
            left_elbow = [landmarks[14].x, landmarks[14].y]     # RIGHT_ELBOW (mirrored)
            left_wrist = [landmarks[16].x, landmarks[16].y]     # RIGHT_WRIST (mirrored)

            right_shoulder = [landmarks[11].x, landmarks[11].y]  # LEFT_SHOULDER (mirrored)
            right_elbow = [landmarks[13].x, landmarks[13].y]     # LEFT_ELBOW (mirrored)
            right_wrist = [landmarks[15].x, landmarks[15].y]     # LEFT_WRIST (mirrored)

            # Calculate angles for both arms
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Check for reps on the left arm - exact Flask logic
            if left_arm_angle <= 30 and self.left_stage == 'down':
                self.left_rep_count += 1
                self.left_stage = 'up'
                self.coach.speak(f"Left arm curl! {self.left_rep_count} left reps")

            if left_arm_angle >= 160:
                self.left_stage = 'down'

            # Check for reps on the right arm - exact Flask logic
            if right_arm_angle <= 30 and self.right_stage == 'down':
                self.right_rep_count += 1
                self.right_stage = 'up'
                self.coach.speak(f"Right arm curl! {self.right_rep_count} right reps")

            if right_arm_angle >= 160:
                self.right_stage = 'down'
                
            # Total reps
            self.rep_count = self.left_rep_count + self.right_rep_count
                
        except Exception as e:
            pass
            
    def count_hammer_curls(self, landmarks):
        try:
            # Same as bicep curls but with different threshold like Flask app
            right_shoulder = [landmarks[11].x, landmarks[11].y]  # LEFT_SHOULDER
            right_elbow = [landmarks[13].x, landmarks[13].y]     # LEFT_ELBOW
            right_wrist = [landmarks[15].x, landmarks[15].y]     # LEFT_WRIST

            left_shoulder = [landmarks[12].x, landmarks[12].y]   # RIGHT_SHOULDER
            left_elbow = [landmarks[14].x, landmarks[14].y]      # RIGHT_ELBOW
            left_wrist = [landmarks[16].x, landmarks[16].y]      # RIGHT_WRIST

            # Calculate angles for both arms
            left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Define the hammer curl angle threshold like Flask app
            curl_angle_threshold = 90

            # Check for reps on the left arm
            if left_arm_angle <= curl_angle_threshold and self.left_stage == 'down':
                self.left_rep_count += 1
                self.left_stage = 'up'
                self.coach.speak(f"Left hammer curl! {self.left_rep_count} left reps")

            if left_arm_angle >= 160:
                self.left_stage = 'down'

            # Check for reps on the right arm
            if right_arm_angle <= curl_angle_threshold and self.right_stage == 'down':
                self.right_rep_count += 1
                self.right_stage = 'up'
                self.coach.speak(f"Right hammer curl! {self.right_rep_count} right reps")

            if right_arm_angle >= 160:
                self.right_stage = 'down'
                
            # Total reps
            self.rep_count = self.left_rep_count + self.right_rep_count
                
        except Exception as e:
            pass
            
    def count_side_raises(self, landmarks):
        try:
            # Get shoulder, elbow landmarks for side raise detection
            left_shoulder = [landmarks[11].x, landmarks[11].y]   # LEFT_SHOULDER
            left_elbow = [landmarks[13].x, landmarks[13].y]      # LEFT_ELBOW
            right_shoulder = [landmarks[12].x, landmarks[12].y]  # RIGHT_SHOULDER  
            right_elbow = [landmarks[14].x, landmarks[14].y]     # RIGHT_ELBOW
            
            # Calculate elevation angle for each arm relative to shoulder
            # For side raises, we want the angle between shoulder and elbow relative to horizontal
            
            # Left arm elevation (0° = horizontal, 90° = straight up, -90° = straight down)
            left_elevation = math.degrees(math.atan2(
                left_shoulder[1] - left_elbow[1],  # y difference (note: inverted due to screen coordinates)
                abs(left_elbow[0] - left_shoulder[0])  # x difference
            ))
            
            # Right arm elevation  
            right_elevation = math.degrees(math.atan2(
                right_shoulder[1] - right_elbow[1],  # y difference (note: inverted due to screen coordinates)
                abs(right_elbow[0] - right_shoulder[0])  # x difference
            ))
            
            # Use the average of both arms for better detection
            avg_elevation = (left_elevation + right_elevation) / 2
            
            # Stage detection for side raises:
            # Down position: arms at side (elevation < 20°)
            # Up position: arms raised to shoulder level (elevation > 60°)
            
            if avg_elevation > 60 and self.stage == "down":
                self.stage = "up"
            elif avg_elevation < 20 and self.stage == "up":
                self.stage = "down"
                self.rep_count += 1
                self.coach.speak(f"Great side raise! {self.rep_count} reps")
                
        except Exception as e:
            pass
            
    def process_frame(self, landmarks):
        if not landmarks:
            return
            
        if self.workout_type == "squats":
            self.count_squats(landmarks.landmark)
        elif self.workout_type == "pushups":
            self.count_pushups(landmarks.landmark)
        elif self.workout_type == "bicep_curls":
            self.count_bicep_curls(landmarks.landmark)
        elif self.workout_type == "hammer_curls":
            self.count_hammer_curls(landmarks.landmark)
        elif self.workout_type == "side_raises":
            self.count_side_raises(landmarks.landmark)
            
    def get_current_reps(self):
        if self.workout_type in ["bicep_curls", "hammer_curls"]:
            return f"L:{self.left_rep_count} R:{self.right_rep_count} Total:{self.rep_count}"
        else:
            return f"{self.rep_count}"
            
    def get_feedback_text(self):
        feedback = []
        if self.workout_type == "squats":
            feedback = ["Keep your back straight", "Go below 100° angle", "Chest up", "Controlled movement"]
        elif self.workout_type == "pushups":
            feedback = ["Keep body straight", "Go below 30° angle", "Full range of motion", "Control both directions"]
        elif self.workout_type in ["bicep_curls", "hammer_curls"]:
            feedback = ["Stable elbows", "Full contraction", "Control the weight", "Both arms equally"]
        elif self.workout_type == "side_raises":
            feedback = ["Shoulder level only", "Control the weight", "45° to 150° range", "Smooth movement"]
        return feedback
        
    def cleanup(self):
        self.coach.stop()
