import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, ChevronLeft, ShieldAlert, Award, Play } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function WorkoutPage() {
  const { workout_name } = useParams();
  const navigate = useNavigate();

  const [repCount, setRepCount] = useState(0);
  const [leftReps, setLeftReps] = useState(0);
  const [rightReps, setRightReps] = useState(0);
  const [stage, setStage] = useState('Calibrating');
  const [feedbackList, setFeedbackList] = useState([]);
  const [active, setActive] = useState(false);
  const [cameraError, setCameraError] = useState(false);
  
  // Track previous rep count for scale popping animation trigger
  const prevRepRef = useRef(0);
  const [popCounter, setPopCounter] = useState(0);

  // Poll real-time reps and stages from local backend
  useEffect(() => {
    let intervalId = null;

    const startPolling = () => {
      intervalId = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE_URL}/api/workout-stats/`);
          const stats = res.data;
          
          if (stats && stats.active) {
            setStage(stats.stage || 'Ready');
            setFeedbackList(stats.feedback || []);
            setActive(true);
            
            // Check if rep count incremented
            const newReps = stats.rep_count || 0;
            if (newReps > prevRepRef.current) {
              setPopCounter(c => c + 1); // trigger scale animation
            }
            setRepCount(newReps);
            prevRepRef.current = newReps;

            setLeftReps(stats.left_rep_count || 0);
            setRightReps(stats.right_rep_count || 0);
          }
        } catch (err) {
          console.error("Error fetching stats", err);
        }
      }, 500);
    };

    startPolling();

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const handleFinish = async () => {
    try {
      // Post final posture results to database
      await axios.post(`${API_BASE_URL}/api/save-posture-analysis/`, {
        exercise_name: workout_name,
        rep_count: repCount,
        stage: 'Complete',
        feedback: feedbackList.slice(-5) // Send last few coaching tips
      });
      navigate('/workouts');
    } catch (err) {
      console.error("Failed to save final posture session", err);
      navigate('/workouts');
    }
  };

  const formattedName = workout_name
    .replace('_', ' ')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');

  return (
    <div className="space-y-6">
      {/* Header Back Button */}
      <div className="flex items-center space-x-4">
        <button
          onClick={handleFinish}
          className="bg-dark-card border border-dark-border hover:bg-dark-border text-gray-300 p-2 rounded-xl transition-all duration-200 cursor-pointer"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>
        <div>
          <h2 className="text-2xl font-black text-white m-0 uppercase tracking-tight">{formattedName} Tracking</h2>
          <p className="text-gray-400 text-xs mt-0.5">MediaPipe real-time skeleton pose correction</p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left/Middle: Live Camera Stream View */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative aspect-video bg-black rounded-3xl border border-dark-border overflow-hidden flex items-center justify-center group shadow-2xl">
            {/* The Streaming Image from Backend */}
            {!cameraError ? (
              <img 
                src={`${API_BASE_URL}/video_feed/${workout_name}/`} 
                alt="Camera Feed"
                onError={() => setCameraError(true)}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="text-center p-6 space-y-3">
                <Camera className="w-12 h-12 text-brand-red mx-auto animate-pulse" />
                <p className="text-sm font-bold text-white">Camera Connection Offline</p>
                <p className="text-xs text-gray-500 max-w-xs mx-auto">
                  Verify your browser camera permissions and make sure no other program is accessing your webcam.
                </p>
              </div>
            )}

            {/* Stage Indicator Overlay */}
            <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-xl flex items-center space-x-2">
              <span className="w-2.5 h-2.5 bg-brand-red rounded-full animate-ping"></span>
              <span className="text-[10px] font-black uppercase tracking-wider text-white">Live AI Correction</span>
            </div>
          </div>

          {/* Tips Info Bar */}
          <div className="bg-dark-card border border-dark-border p-4 rounded-2xl flex items-center space-x-3">
            <ShieldAlert className="w-5 h-5 text-brand-red shrink-0" />
            <p className="text-xs text-gray-400">
              <strong className="text-white">Tips:</strong> Ensure your entire body (shoulders down to ankles) is visible in the frame for accurate landmark estimation.
            </p>
          </div>
        </div>

        {/* Right Sidebar: Real-Time Stats & Feedback */}
        <div className="space-y-6">
          
          {/* Main Counters Panel */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-3xl flex flex-col items-center justify-between text-center relative overflow-hidden shadow-xl">
            <div className="w-full pb-4 border-b border-dark-border/50">
              <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Active Repetitions</span>
              
              {/* Rep Count with Scale-Pop Animation */}
              <div className="relative my-4 flex justify-center">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={popCounter}
                    initial={{ scale: 0.8, opacity: 0.5 }}
                    animate={{ scale: [1, 1.35, 1], opacity: 1 }}
                    transition={{ duration: 0.35, ease: 'easeOut' }}
                    className="text-7xl font-black text-white leading-none tracking-tight select-none"
                  >
                    {repCount}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Left/Right Splits (for bilateral curls, etc.) */}
            {(leftReps > 0 || rightReps > 0) && (
              <div className="w-full grid grid-cols-2 gap-4 py-3 border-b border-dark-border/50">
                <div className="text-center">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Left Arm</span>
                  <p className="text-base font-bold text-white mt-0.5">{leftReps}</p>
                </div>
                <div className="text-center">
                  <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Right Arm</span>
                  <p className="text-base font-bold text-white mt-0.5">{rightReps}</p>
                </div>
              </div>
            )}

            {/* Workout State Stage */}
            <div className="w-full pt-4">
              <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Current Position</span>
              <p className="text-lg font-black text-brand-red uppercase tracking-wider mt-1 select-none">
                {stage}
              </p>
            </div>
          </div>

          {/* Real-time Biomechanics Feedback Tips */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-3xl flex-1 flex flex-col justify-between shadow-xl">
            <div>
              <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block mb-4">AI Biomechanics Feedback</span>
              <div className="space-y-3 max-h-48 overflow-y-auto">
                {feedbackList.length > 0 ? (
                  feedbackList.slice(-4).reverse().map((tip, idx) => (
                    <motion.div 
                      key={idx}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2, delay: idx * 0.05 }}
                      className={`text-xs p-3 rounded-xl border flex items-start space-x-2 ${
                        idx === 0 
                          ? 'bg-brand-red/10 border-brand-red/30 text-white' 
                          : 'bg-dark-bg border-dark-border text-gray-400'
                      }`}
                    >
                      <Award className={`w-4 h-4 shrink-0 mt-0.5 ${idx === 0 ? 'text-brand-red' : 'text-gray-500'}`} />
                      <span>{tip}</span>
                    </motion.div>
                  ))
                ) : (
                  <div className="text-center py-6 text-xs text-gray-500">
                    Waiting for movement calibration data...
                  </div>
                )}
              </div>
            </div>

            <button
              onClick={handleFinish}
              className="mt-6 w-full bg-brand-red hover:bg-brand-red-hover text-white py-3 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer"
            >
              FINISH SESSION
            </button>
          </div>

        </div>

      </div>
    </div>
  );
}
