import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Dumbbell, 
  Utensils, 
  Calculator, 
  MessageSquare, 
  Camera, 
  Flame, 
  Sparkles, 
  TrendingUp, 
  Activity 
} from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function Dashboard({ user }) {
  const [profile, setProfile] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [loadingProfile, setLoadingProfile] = useState(true);
  const [loadingRec, setLoadingRec] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const profileRes = await axios.get(`${API_BASE_URL}/api/profile-setup/`);
        if (profileRes.data.has_profile) {
          setProfile(profileRes.data);
        } else {
          navigate('/profile');
        }
      } catch (err) {
        console.error("Error fetching profile", err);
      } finally {
        setLoadingProfile(false);
      }

      try {
        const recRes = await axios.get(`${API_BASE_URL}/api/generate-recommendation/`);
        if (recRes.data.status === 'success') {
          setRecommendation(recRes.data);
        }
      } catch (err) {
        console.error("Error fetching recommendation", err);
      } finally {
        setLoadingRec(false);
      }
    };

    fetchDashboardData();
  }, [navigate]);

  const modules = [
    { 
      name: 'Pose Correction', 
      desc: 'Real-time AI posture evaluation for squats, curls, and pushups.', 
      path: '/workouts', 
      icon: Dumbbell,
      color: 'from-red-500/5 to-transparent' 
    },
    { 
      name: 'Calorie Tracker', 
      desc: 'Voice-activated food logging, nutrient macro rings, and calorie tracking.', 
      path: '/diet', 
      icon: Utensils,
      color: 'from-orange-500/5 to-transparent' 
    },
    { 
      name: 'AI Coach Chat', 
      desc: 'Intent-aware chatbot trainer providing custom biomechanics feedback.', 
      path: '/chat', 
      icon: MessageSquare,
      color: 'from-blue-500/5 to-transparent' 
    },
    { 
      name: '1RM Calculator', 
      desc: 'Determine your strength thresholds and workload intensities.', 
      path: '/1rm', 
      icon: Calculator,
      color: 'from-purple-500/5 to-transparent' 
    },
    { 
      name: 'Carb Cycling', 
      desc: 'Adaptive high/low carb protocols matching your workout volume.', 
      path: '/carb-cycling', 
      icon: Flame,
      color: 'from-yellow-500/5 to-transparent' 
    },
    { 
      name: 'Body Vision', 
      desc: 'AI physique development analysis & skeletal balance mapping.', 
      path: '/body-vision', 
      icon: Camera,
      color: 'from-emerald-500/5 to-transparent' 
    },
  ];

  if (loadingProfile) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-500 text-sm font-semibold">Loading your fitness dashboard...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Premium Hero Banner */}
      <div className="relative bg-gradient-to-r from-slate-900 via-slate-800 to-slate-950 rounded-3xl p-6 md:p-8 text-white overflow-hidden shadow-lg border border-slate-850">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-20 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-brand-red via-transparent to-transparent pointer-events-none"></div>
        <div className="absolute left-10 -bottom-10 w-40 h-40 bg-brand-red/10 rounded-full blur-3xl pointer-events-none"></div>
        
        <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center space-x-2 bg-brand-red/20 text-brand-red border border-brand-red/30 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest">
              <span className="w-1.5 h-1.5 bg-brand-red rounded-full animate-ping"></span>
              <span>ATHLETE PROFILE ACTIVE</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-black tracking-tight text-white m-0">
              Welcome back, {user?.username}!
            </h2>
            <p className="text-slate-400 text-sm max-w-xl font-medium leading-relaxed">
              Your biometric scanners and training engines are calibrated. Ready to stay hard? Review your telemetry metrics and AI split recommendations.
            </p>
          </div>
          
          {profile && (
            <div className="flex items-center space-x-3 bg-white/5 border border-white/10 backdrop-blur-md px-5 py-3 rounded-2xl self-start md:self-auto shrink-0 shadow-inner">
              <Activity className="w-5 h-5 text-brand-red animate-pulse" />
              <div>
                <span className="text-[9px] font-black uppercase tracking-widest text-slate-400 block">CURRENT GOAL</span>
                <span className="text-sm font-black text-white uppercase mt-0.5 block">{profile.primary_goal?.replace('_', ' ')}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Telemetry Biometrics Row */}
      {profile && (
        <div>
          <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4">Biometric Telemetry</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            
            {/* Weight Card */}
            <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between shadow-sm hover:shadow-md transition-all duration-300 group">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Bodyweight</span>
                <div className="p-2 rounded-xl bg-orange-50 text-orange-500 group-hover:bg-orange-500 group-hover:text-white transition-all duration-300">
                  <TrendingUp className="w-4 h-4" />
                </div>
              </div>
              <div className="flex items-baseline space-x-1.5 mt-4">
                <span className="text-3xl font-black text-slate-900">{profile.weight}</span>
                <span className="text-xs text-slate-400 font-bold">kg</span>
              </div>
              <span className="text-[9px] text-slate-400 font-semibold mt-2 block">Logged athlete profile weight</span>
            </div>

            {/* Height Card */}
            <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between shadow-sm hover:shadow-md transition-all duration-300 group">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Height</span>
                <div className="p-2 rounded-xl bg-blue-50 text-blue-500 group-hover:bg-blue-500 group-hover:text-white transition-all duration-300">
                  <Activity className="w-4 h-4" />
                </div>
              </div>
              <div className="flex items-baseline space-x-1.5 mt-4">
                <span className="text-3xl font-black text-slate-900">{profile.height}</span>
                <span className="text-xs text-slate-400 font-bold">cm</span>
              </div>
              <span className="text-[9px] text-slate-400 font-semibold mt-2 block">Standard standing height</span>
            </div>

            {/* BMI Card */}
            <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between shadow-sm hover:shadow-md transition-all duration-300 group">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Body Mass Index</span>
                <div className="p-2 rounded-xl bg-emerald-50 text-emerald-500 group-hover:bg-emerald-500 group-hover:text-white transition-all duration-300">
                  <Activity className="w-4 h-4" />
                </div>
              </div>
              <div className="flex flex-col mt-4 space-y-1">
                <div className="flex items-baseline space-x-1.5">
                  <span className="text-3xl font-black text-slate-900">{profile.bmi}</span>
                </div>
                <div className="w-fit">
                  <span className="text-[8px] font-black text-brand-red bg-brand-red/10 border border-brand-red/15 px-2.5 py-0.5 rounded-full uppercase tracking-wider">
                    {profile.bmi_category}
                  </span>
                </div>
              </div>
            </div>

            {/* Daily Calorie Budget Card */}
            <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between shadow-sm hover:shadow-md transition-all duration-300 group">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Calorie Budget</span>
                <div className="p-2 rounded-xl bg-red-50 text-brand-red group-hover:bg-brand-red group-hover:text-white transition-all duration-300">
                  <Flame className="w-4 h-4" />
                </div>
              </div>
              <div className="flex items-baseline space-x-1.5 mt-4">
                <span className="text-3xl font-black text-slate-900">{profile.calories_per_day || 2500}</span>
                <span className="text-xs text-slate-400 font-bold">kcal</span>
              </div>
              <span className="text-[9px] text-slate-400 font-semibold mt-2 block">Active metabolic target</span>
            </div>

          </div>
        </div>
      )}

      {/* Feature Modules Grid */}
      <div>
        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4">Core Training Suites</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((m) => {
            const Icon = m.icon;
            
            // Generate clean technology badges based on path
            let techTag = "AI Telemetry";
            if (m.path === "/workouts") techTag = "MediaPipe Vision";
            else if (m.path === "/diet") techTag = "Voice Speech Recognition";
            else if (m.path === "/chat") techTag = "Cognitive AI Coach";
            else if (m.path === "/carb-cycling") techTag = "Metabolic Engine";
            else if (m.path === "/body-vision") techTag = "Physique Vision Scanner";

            return (
              <motion.div
                key={m.name}
                whileHover={{ y: -6, scale: 1.01 }}
                className="relative bg-dark-card border border-dark-border hover:border-dark-border-hover p-6 rounded-3xl shadow-sm cursor-pointer overflow-hidden transition-all duration-300 group"
                onClick={() => navigate(m.path)}
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${m.color} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

                <div className="relative z-10 flex flex-col justify-between h-full space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="bg-brand-red/10 text-brand-red p-3 rounded-xl w-fit flex items-center justify-center group-hover:bg-brand-red group-hover:text-white transition-all duration-300">
                        <Icon className="w-5 h-5" />
                      </div>
                      <span className="text-[8px] font-black uppercase tracking-wider bg-slate-100 text-slate-500 border border-slate-200/50 px-2 py-0.5 rounded-md">
                        {techTag}
                      </span>
                    </div>
                    
                    <h4 className="text-lg font-black text-slate-900 mb-2">{m.name}</h4>
                    <p className="text-slate-500 text-xs leading-relaxed font-semibold">{m.desc}</p>
                  </div>
                  
                  <div className="text-brand-red text-xs font-bold flex items-center space-x-1 group-hover:translate-x-1 transition-transform duration-300 pt-2 border-t border-slate-100/50">
                    <span>LAUNCH ENGINE</span>
                    <span>→</span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Adaptive Recommendation Panel */}
      <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-sm">
        <div className="flex items-center justify-between mb-6 border-b border-dark-border pb-4">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-brand-red animate-pulse" />
            <h3 className="text-xs font-black text-slate-900 m-0 uppercase tracking-widest">AI Workout Recommendation</h3>
          </div>
          <span className="text-[8px] font-black bg-brand-red/10 text-brand-red border border-brand-red/15 px-2.5 py-1 rounded-md uppercase tracking-wider">
            Adaptive Protocol
          </span>
        </div>

        {loadingRec ? (
          <div className="py-12 text-center text-slate-450 text-xs font-semibold flex flex-col items-center justify-center space-y-3">
            <div className="w-8 h-8 border-2 border-brand-red border-t-transparent rounded-full animate-spin"></div>
            <span>Formulating your personalized workout split...</span>
          </div>
        ) : recommendation ? (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-dark-border pb-4 gap-4 sm:gap-0">
              <div>
                <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Target Split Focus</p>
                <h4 className="text-xl font-black text-slate-900 mt-1 uppercase leading-none">
                  {(recommendation.focus || recommendation.focus_areas) ? (recommendation.focus || recommendation.focus_areas).join(" & ") : "Full Body Conditioning"}
                </h4>
              </div>
              <div className="flex space-x-3">
                <div className="bg-white px-4 py-2.5 rounded-xl border border-dark-border text-center shadow-sm">
                  <span className="text-[8px] text-slate-400 uppercase font-black tracking-widest block">Difficulty</span>
                  <p className="text-xs font-black text-brand-red mt-1 uppercase leading-none">{recommendation.difficulty || 'Intermediate'}</p>
                </div>
                <div className="bg-white px-4 py-2.5 rounded-xl border border-dark-border text-center shadow-sm">
                  <span className="text-[8px] text-slate-400 uppercase font-black tracking-widest block">Est. Duration</span>
                  <p className="text-xs font-black text-slate-900 mt-1 leading-none">{(recommendation.duration || recommendation.estimated_duration || 45)} mins</p>
                </div>
              </div>
            </div>

            <div>
              <p className="text-[9px] font-bold text-slate-400 uppercase tracking-widest mb-3">Recommended Exercises</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {recommendation.routine ? (
                  Array.isArray(recommendation.routine) ? (
                    recommendation.routine.map((ex, idx) => (
                      <div key={idx} className="bg-white border border-dark-border p-4 rounded-2xl flex items-center justify-between shadow-sm hover:border-slate-300 transition-colors duration-250">
                        <div className="flex items-center space-x-3">
                          <div className="bg-brand-red/10 text-brand-red w-8 h-8 rounded-lg flex items-center justify-center font-black text-xs">
                            {idx + 1}
                          </div>
                          <div>
                            <p className="text-sm font-bold text-slate-900">{ex.exercise}</p>
                            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider block mt-0.5">{ex.muscle_group}</span>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-black text-slate-900">{ex.sets} x {ex.reps}</p>
                          <span className="text-[9px] text-brand-red font-bold uppercase tracking-wider block mt-0.5">{ex.weight}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    Object.entries(recommendation.routine).map(([day, exercises], idx) => (
                      <div key={idx} className="bg-white border border-dark-border p-5 rounded-2xl shadow-sm hover:border-slate-300 transition-colors duration-250 space-y-2 col-span-1 md:col-span-2">
                        <div className="flex items-center space-x-3">
                          <div className="bg-brand-red/10 text-brand-red px-3 py-1 rounded-lg font-black text-[10px] uppercase tracking-wider">
                            {day}
                          </div>
                        </div>
                        <p className="text-slate-650 text-sm font-semibold leading-relaxed pl-1">
                          {exercises}
                        </p>
                      </div>
                    ))
                  )
                ) : (
                  recommendation.recommended_exercises && (
                    Array.isArray(recommendation.recommended_exercises) ? (
                      recommendation.recommended_exercises.map((ex, idx) => (
                        <div key={idx} className="bg-white border border-dark-border p-4 rounded-2xl flex items-center justify-between shadow-sm hover:border-slate-300 transition-colors duration-250">
                          <div className="flex items-center space-x-3">
                            <div className="bg-brand-red/10 text-brand-red w-8 h-8 rounded-lg flex items-center justify-center font-black text-xs">
                              {idx + 1}
                            </div>
                            <div>
                              <p className="text-sm font-bold text-slate-900">{ex.name}</p>
                              <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider block mt-0.5">{ex.muscle}</span>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-black text-slate-900">{ex.sets} x {ex.reps}</p>
                            <span className="text-[9px] text-brand-red font-bold uppercase tracking-wider block mt-0.5">{ex.weight || 'Bodyweight'}</span>
                          </div>
                        </div>
                      ))
                    ) : (
                      Object.entries(recommendation.recommended_exercises).map(([day, exercises], idx) => (
                        <div key={idx} className="bg-white border border-dark-border p-5 rounded-2xl shadow-sm hover:border-slate-300 transition-colors duration-250 space-y-2 col-span-1 md:col-span-2">
                          <div className="flex items-center space-x-3">
                            <div className="bg-brand-red/10 text-brand-red px-3 py-1 rounded-lg font-black text-[10px] uppercase tracking-wider">
                              {day}
                            </div>
                          </div>
                          <p className="text-slate-650 text-sm font-semibold leading-relaxed pl-1">
                            {exercises}
                          </p>
                        </div>
                      ))
                    )
                  )
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center text-slate-400 text-xs font-semibold">
            No recommendation routine available. Create your profile details to generate daily splits.
          </div>
        )}
      </div>
    </div>
  );
}
