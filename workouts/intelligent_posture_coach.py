"""
AI-Powered Intelligent Posture Coach
Uses LLM reasoning with computer vision data for natural, adaptive feedback
"""

import cv2
import numpy as np
import time
import threading
import json
import requests
import os
from queue import Queue
from dotenv import load_dotenv

load_dotenv()

# Optional text-to-speech for voice feedback
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class IntelligentPostureCoach:
    def __init__(self, exercise_type, user_profile=None):
        self.exercise_type = exercise_type
        self.user_profile = user_profile or {}
        
        # AI-powered analysis state
        self.session_state = "initializing"  # initializing -> positioning -> ready -> coaching -> complete
        self.conversation_history = []
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # Analyze every 2 seconds for intelligent feedback
        
        # Computer vision data buffer
        self.pose_data_buffer = []
        self.current_pose_analysis = {}
        
        # LLM Configuration
        self.setup_llm_connection()
        
        # Voice feedback system
        self.setup_voice_feedback()
        
        # Coaching context and memory
        self.coaching_memory = {
            'user_improvements': [],
            'common_issues': [],
            'positive_feedback_given': [],
            'corrections_made': []
        }
        
        print(f"IntelligentPostureCoach: AI-powered coaching enabled for {exercise_type}")
    
    def setup_llm_connection(self):
        """Setup connection to Ollama for intelligent analysis"""
        # Use only Ollama for free local LLM inference
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default model
        
        # Test connection
        self.llm_available = self.test_ollama_connection()
        if self.llm_available:
            print(f"IntelligentPostureCoach: Ollama connected successfully with model {self.ollama_model}")
        else:
            print("IntelligentPostureCoach: WARNING - Ollama not available. Make sure Ollama is running and llama3.2 model is installed.")
    
    def test_ollama_connection(self):
        """Test if Ollama is available and responsive"""
        try:
            # Quick test with Ollama
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": "Hi", "stream": False},
                timeout=5
            )
            if response.status_code == 200:
                return True
            else:
                print(f"Ollama connection failed with status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Ollama connection error: {e}")
            return False
    
    def setup_voice_feedback(self):
        """Initialize intelligent voice system"""
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                
                # Set a friendly voice
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            self.tts_engine.setProperty('voice', voice.id)
                            break
                
                self.voice_queue = Queue()
                self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
                self.voice_thread.start()
                self.voice_enabled = True
                print("IntelligentPostureCoach: Intelligent voice coaching enabled")
            except Exception as e:
                print(f"IntelligentPostureCoach: Voice system failed: {e}")
                self.voice_enabled = False
        else:
            self.voice_enabled = False
    
    def _voice_worker(self):
        """Background worker for voice feedback"""
        while True:
            try:
                message = self.voice_queue.get(timeout=1)
                if message is None:
                    break
                if self.voice_enabled and message.strip():
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
                self.voice_queue.task_done()
            except:
                continue
    
    def analyze_pose_with_ai(self, landmarks):
        """AI-powered pose analysis using LLM reasoning"""
        current_time = time.time()
        
        # Only analyze every few seconds to avoid overwhelming
        if current_time - self.last_analysis_time < self.analysis_interval:
            return
        
        if not self.llm_available:
            return  # Fallback to basic analysis if needed
        
        self.last_analysis_time = current_time
        
        # Extract meaningful pose data
        pose_metrics = self.extract_pose_metrics(landmarks)
        
        # Build intelligent context for LLM
        analysis_prompt = self.build_analysis_prompt(pose_metrics)
        
        # Get AI analysis
        ai_response = self.query_llm_for_analysis(analysis_prompt)
        
        # Process AI response
        self.process_ai_feedback(ai_response)
    
    def extract_pose_metrics(self, landmarks):
        """Extract meaningful metrics from pose landmarks"""
        try:
            # Key body points
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Calculate meaningful metrics
            metrics = {
                # Body positioning
                'shoulder_width': abs(right_shoulder.x - left_shoulder.x),
                'hip_width': abs(right_hip.x - left_hip.x),
                'body_height_ratio': abs(nose.y - left_hip.y),
                
                # Stance analysis
                'leg_separation': abs(right_ankle.x - left_ankle.x),
                'body_center_x': (left_shoulder.x + right_shoulder.x) / 2,
                'body_center_y': (left_shoulder.y + right_shoulder.y) / 2,
                
                # Posture indicators
                'torso_alignment': abs((left_shoulder.x + right_shoulder.x) / 2 - (left_hip.x + right_hip.x) / 2),
                'head_position': nose.x - (left_shoulder.x + right_shoulder.x) / 2,
                
                # Leg position (sitting vs standing)
                'hip_knee_distance': (left_knee.y + right_knee.y) / 2 - (left_hip.y + right_hip.y) / 2,
                'knee_ankle_distance': (left_ankle.y + right_ankle.y) / 2 - (left_knee.y + right_knee.y) / 2,
                
                # Visibility checks
                'feet_visible': left_ankle.y > 0.8 and right_ankle.y > 0.8,
                'full_body_visible': nose.y < 0.2 and left_ankle.y > 0.8,
                
                # Frame positioning
                'too_close': metrics['shoulder_width'] > 0.4,
                'too_far': metrics['shoulder_width'] < 0.1,
            }
            
            # Calculate leg angles for sitting detection
            left_leg_angle = self.calculate_angle(
                [left_hip.x, left_hip.y],
                [left_knee.x, left_knee.y],
                [left_ankle.x, left_ankle.y]
            )
            right_leg_angle = self.calculate_angle(
                [right_hip.x, right_hip.y],
                [right_knee.x, right_knee.y],
                [right_ankle.x, right_ankle.y]
            )
            
            metrics['avg_leg_angle'] = (left_leg_angle + right_leg_angle) / 2
            metrics['likely_standing'] = metrics['avg_leg_angle'] > 140 and metrics['hip_knee_distance'] > 0.1
            
            self.current_pose_analysis = metrics
            return metrics
            
        except Exception as e:
            print(f"IntelligentPostureCoach: Error extracting pose metrics: {e}")
            return {}
    
    def build_analysis_prompt(self, pose_metrics):
        """Build intelligent prompt for LLM analysis"""
        
        # Context about the exercise
        exercise_context = {
            'bicep_curls': 'standing exercise requiring good posture, elbows at sides, facing camera for bilateral arm tracking',
            'hammer_curls': 'standing exercise requiring good posture, elbows at sides, facing camera for bilateral arm tracking', 
            'squats': 'standing exercise requiring full body visibility, proper stance width, upright posture',
            'pushups': 'floor exercise, user may be lying down, upper body focus, close to camera is OK',
            'side_raises': 'standing exercise requiring upright posture, arms at sides, good shoulder visibility'
        }
        
        current_context = exercise_context.get(self.exercise_type, 'fitness exercise requiring good form')
        
        prompt = f"""You are an intelligent AI fitness coach analyzing a user's posture for {self.exercise_type}. 

EXERCISE CONTEXT: {current_context}

CURRENT SESSION STATE: {self.session_state}
- initializing: Just started, getting user positioned
- positioning: User making adjustments based on feedback  
- ready: User is properly positioned, ready to begin workout
- coaching: Actively coaching during exercise
- complete: Session finished

POSE ANALYSIS DATA:
- Shoulder width: {pose_metrics.get('shoulder_width', 0):.3f} (0.15+ = good camera facing)
- Hip width: {pose_metrics.get('hip_width', 0):.3f}
- Body height in frame: {pose_metrics.get('body_height_ratio', 0):.3f}
- Leg separation: {pose_metrics.get('leg_separation', 0):.3f}
- Average leg angle: {pose_metrics.get('avg_leg_angle', 0):.1f}° (140+ = standing)
- Hip-knee distance: {pose_metrics.get('hip_knee_distance', 0):.3f} (0.1+ = legs extended)
- Knee-ankle distance: {pose_metrics.get('knee_ankle_distance', 0):.3f}
- Feet visible: {pose_metrics.get('feet_visible', False)}
- Full body visible: {pose_metrics.get('full_body_visible', False)}
- Likely standing: {pose_metrics.get('likely_standing', False)}
- Head alignment: {pose_metrics.get('head_position', 0):.3f} (close to 0 = centered)

USER COACHING HISTORY:
Recent feedback: {self.coaching_memory.get('corrections_made', [])[-3:]}
Improvements made: {self.coaching_memory.get('user_improvements', [])[-3:]}

TASK: Analyze the user's current posture and provide intelligent feedback as a supportive fitness coach.

RESPONSE FORMAT (JSON):
{{
    "posture_assessment": "excellent|good|needs_adjustment|poor",
    "primary_issue": "sitting|facing_away|too_close|too_far|stance|alignment|none",
    "feedback_message": "Natural, encouraging message for the user",
    "voice_command": "Clear, concise voice instruction (if correction needed)",
    "next_state": "positioning|ready|coaching",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of assessment"
}}

COACHING GUIDELINES:
1. Be encouraging and supportive, not robotic
2. Acknowledge improvements: "Great! You're in perfect position now"
3. Give clear, actionable feedback: "Please stand up and face the camera"
4. Transition naturally: "Perfect posture! Let's start your workout"
5. Don't repeat the same feedback - be adaptive
6. Consider the exercise requirements specifically

Analyze the current posture state and provide intelligent feedback:"""

        return prompt
    
    def query_llm_for_analysis(self, prompt):
        """Query Ollama for intelligent posture analysis"""
        if not self.llm_available:
            return self.get_fallback_message()
            
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,  # Slightly higher for more natural responses
                    "top_p": 0.9,
                    "max_tokens": 250,
                    "num_ctx": 2048  # Context length
                }
            }
            
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=8  # Reasonable timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                if ai_response:
                    return ai_response
                else:
                    return self.get_fallback_message()
            else:
                print(f"Ollama API returned status {response.status_code}")
                return self.get_fallback_message()
            
        except requests.exceptions.Timeout:
            print("Ollama request timed out")
            return self.get_fallback_message()
        except requests.exceptions.RequestException as e:
            print(f"Ollama request failed: {e}")
            return self.get_fallback_message()
        except Exception as e:
            print(f"Ollama processing error: {e}")
            return self.get_fallback_message()
    
    def get_fallback_message(self):
        """Provide fallback messages when Ollama is not available"""
        fallback_messages = {
            "initializing": "Welcome! Please position yourself in front of the camera.",
            "positioning": "Great! I can see you. Let's get you positioned for the exercise.",
            "ready": "Perfect position! You're ready to begin your workout.",
            "coaching": "Keep up the great work! Focus on your form.",
            "complete": "Excellent session! Well done on completing your workout."
        }
        return fallback_messages.get(self.session_state, "Keep going! You're doing great!")
    
    def process_ai_feedback(self, ai_response):
        """Process and act on AI feedback"""
        if not ai_response:
            return
        
        try:
            # Extract JSON response from AI
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                feedback_data = json.loads(json_match.group())
            else:
                # Fallback parsing
                feedback_data = self.parse_natural_response(ai_response)
            
            # Update session state based on AI assessment
            new_state = feedback_data.get('next_state', self.session_state)
            if new_state != self.session_state:
                print(f"State transition: {self.session_state} -> {new_state}")
                self.session_state = new_state
            
            # Store feedback in memory
            self.update_coaching_memory(feedback_data)
            
            # Provide voice feedback if needed
            voice_command = feedback_data.get('voice_command', '')
            if voice_command and self.voice_enabled:
                self.speak_feedback(voice_command)
            
            # Store current feedback for display
            self.current_feedback = {
                'assessment': feedback_data.get('posture_assessment', 'analyzing'),
                'message': feedback_data.get('feedback_message', 'Analyzing your posture...'),
                'confidence': feedback_data.get('confidence', 0.5),
                'reasoning': feedback_data.get('reasoning', '')
            }
            
        except Exception as e:
            print(f"Error processing AI feedback: {e}")
    
    def parse_natural_response(self, response):
        """Parse natural language response if JSON extraction fails"""
        # Simple fallback parsing
        assessment = "analyzing"
        if "excellent" in response.lower() or "perfect" in response.lower():
            assessment = "excellent"
        elif "good" in response.lower():
            assessment = "good"
        elif "needs" in response.lower() or "adjust" in response.lower():
            assessment = "needs_adjustment"
        
        return {
            'posture_assessment': assessment,
            'feedback_message': response[:100] + "..." if len(response) > 100 else response,
            'voice_command': '',
            'next_state': 'positioning',
            'confidence': 0.5
        }
    
    def update_coaching_memory(self, feedback_data):
        """Update coaching memory for adaptive feedback"""
        assessment = feedback_data.get('posture_assessment', '')
        issue = feedback_data.get('primary_issue', '')
        
        # Track improvements
        if assessment in ['excellent', 'good'] and self.session_state == 'positioning':
            self.coaching_memory['user_improvements'].append(f"Achieved {assessment} posture")
        
        # Track corrections made
        if issue and issue != 'none':
            self.coaching_memory['corrections_made'].append(issue)
        
        # Keep memory manageable
        for key in self.coaching_memory:
            if len(self.coaching_memory[key]) > 10:
                self.coaching_memory[key] = self.coaching_memory[key][-5:]
    
    def speak_feedback(self, message):
        """Queue intelligent voice feedback"""
        if self.voice_enabled and message.strip():
            try:
                self.voice_queue.put(message, block=False)
            except:
                pass
    
    def get_display_feedback(self):
        """Get feedback for display overlay"""
        if not hasattr(self, 'current_feedback'):
            return {
                'status': 'Analyzing...',
                'message': 'Getting ready...',
                'color': (255, 255, 0)  # Yellow
            }
        
        feedback = self.current_feedback
        assessment = feedback.get('assessment', 'analyzing')
        
        # Color coding based on AI assessment
        color_map = {
            'excellent': (0, 255, 0),      # Green
            'good': (0, 255, 150),         # Light green
            'needs_adjustment': (0, 165, 255),  # Orange
            'poor': (0, 100, 255),         # Red
            'analyzing': (255, 255, 0)     # Yellow
        }
        
        status_map = {
            'excellent': '✅ Perfect Posture!',
            'good': '✅ Good Position',
            'needs_adjustment': '⚠️ Please Adjust',
            'poor': '❌ Needs Correction',
            'analyzing': '🤖 Analyzing...'
        }
        
        return {
            'status': status_map.get(assessment, '🤖 Analyzing...'),
            'message': feedback.get('message', 'Analyzing your posture...'),
            'color': color_map.get(assessment, (255, 255, 0)),
            'confidence': feedback.get('confidence', 0.5)
        }
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle
    
    def is_ready_for_workout(self):
        """Check if user is ready to start workout"""
        return self.session_state in ['ready', 'coaching']
    
    def transition_to_workout_mode(self):
        """Smoothly transition to workout coaching"""
        if self.session_state == 'ready':
            self.session_state = 'coaching'
            self.speak_feedback(f"Perfect! Let's begin your {self.exercise_type.replace('_', ' ')} workout!")
    
    def cleanup(self):
        """Clean up resources"""
        if self.voice_enabled:
            try:
                self.voice_queue.put(None)
            except:
                pass