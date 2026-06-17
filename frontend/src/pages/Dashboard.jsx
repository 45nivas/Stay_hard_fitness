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
          // If no profile, redirect to setup
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
      color: 'from-red-500/20 to-transparent' 
    },
    { 
      name: 'Calorie Tracker', 
      desc: 'Voice-activated food logging, nutrient macro rings, and calorie tracking.', 
      path: '/diet', 
      icon: Utensils,
      color: 'from-orange-500/20 to-transparent' 
    },
    { 
      name: 'AI Coach Chat', 
      desc: 'Intent-aware chatbot trainer providing custom biomechanics feedback.', 
      path: '/chat', 
      icon: MessageSquare,
      color: 'from-blue-500/20 to-transparent' 
    },
    { 
      name: '1RM Calculator', 
      desc: 'Determine your strength thresholds and workload intensities.', 
      path: '/1rm', 
      icon: Calculator,
      color: 'from-purple-500/20 to-transparent' 
    },
    { 
      name: 'Carb Cycling', 
      desc: 'Adaptive high/low carb protocols matching your workout volume.', 
      path: '/carb-cycling', 
      icon: Flame,
      color: 'from-yellow-500/20 to-transparent' 
    },
    { 
      name: 'Body Vision', 
      desc: 'AI physique development analysis & skeletal balance mapping.', 
      path: '/body-vision', 
      icon: Camera,
      color: 'from-emerald-500/20 to-transparent' 
    },
  ];

  if (loadingProfile) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-400 text-sm font-semibold">Loading your fitness dashboard...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Welcome Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
        <div>
          <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">
            Welcome back, {user?.username}!
          </h2>
          <p className="text-gray-400 text-sm mt-1">Ready to stay hard? Here is your daily fitness overview.</p>
        </div>
        
        {profile && (
          <div className="flex items-center space-x-2 bg-dark-card border border-dark-border px-4 py-2 rounded-xl">
            <Activity className="w-5 h-5 text-brand-red" />
            <span className="text-xs font-bold uppercase tracking-wider text-gray-300">Goal:</span>
            <span className="text-xs font-bold text-white uppercase">{profile.primary_goal?.replace('_', ' ')}</span>
          </div>
        )}
      </div>

      {/* Metrics Row */}
      {profile && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Weight</span>
            <div className="flex items-baseline space-x-1 mt-2">
              <span className="text-2xl font-black text-white">{profile.weight}</span>
              <span className="text-xs text-gray-400 font-bold">kg</span>
            </div>
          </div>

          <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Height</span>
            <div className="flex items-baseline space-x-1 mt-2">
              <span className="text-2xl font-black text-white">{profile.height}</span>
              <span className="text-xs text-gray-400 font-bold">cm</span>
            </div>
          </div>

          <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Body Mass Index (BMI)</span>
            <div className="flex items-baseline space-x-2 mt-2">
              <span className="text-2xl font-black text-white">{profile.bmi}</span>
              <span className="text-xs font-bold text-brand-red bg-brand-red/10 px-2 py-0.5 rounded-full uppercase tracking-wider">
                {profile.bmi_category}
              </span>
            </div>
          </div>

          <div className="bg-dark-card border border-dark-border p-5 rounded-2xl flex flex-col justify-between">
            <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Daily Target Calories</span>
            <div className="flex items-baseline space-x-1 mt-2">
              <span className="text-2xl font-black text-white">{profile.calories_per_day || 2500}</span>
              <span className="text-xs text-gray-400 font-bold">kcal</span>
            </div>
          </div>
        </div>
      )}

      {/* Feature Modules Grid */}
      <div>
        <h3 className="text-lg font-bold text-gray-300 mb-4 uppercase tracking-wider">Feature Modules</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {modules.map((m, idx) => {
            const Icon = m.icon;
            return (
              <motion.div
                key={m.name}
                whileHover={{ y: -6, scale: 1.02 }}
                className="relative bg-dark-card border border-dark-border hover:border-dark-border-hover p-6 rounded-2xl shadow-lg cursor-pointer overflow-hidden transition-all duration-300 group"
                onClick={() => navigate(m.path)}
              >
                {/* Accent glow on hover */}
                <div className={`absolute inset-0 bg-gradient-to-br ${m.color} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

                <div className="relative z-10 flex flex-col justify-between h-full">
                  <div className="mb-4">
                    <div className="bg-brand-red/10 text-brand-red p-3 rounded-xl w-fit flex items-center justify-center mb-4 group-hover:bg-brand-red group-hover:text-white transition-all duration-300">
                      <Icon className="w-6 h-6" />
                    </div>
                    <h4 className="text-lg font-bold text-white mb-2">{m.name}</h4>
                    <p className="text-gray-400 text-xs leading-relaxed">{m.desc}</p>
                  </div>
                  <div className="text-brand-red text-xs font-bold flex items-center space-x-1 group-hover:translate-x-1 transition-transform duration-300 mt-2">
                    <span>Enter Module</span>
                    <span>→</span>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Adaptive Recommendation Panel */}
      <div className="bg-dark-card border border-dark-border p-6 rounded-2xl shadow-lg">
        <div className="flex items-center space-x-2 mb-6">
          <Sparkles className="w-5 h-5 text-brand-red animate-pulse" />
          <h3 className="text-lg font-bold text-white m-0 uppercase tracking-wider">AI Workout Recommendation</h3>
        </div>

        {loadingRec ? (
          <div className="py-8 text-center text-gray-500 text-sm font-semibold">
            Formulating your personalized workout split...
          </div>
        ) : recommendation ? (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-dark-border pb-4 gap-4 sm:gap-0">
              <div>
                <p className="text-sm font-bold text-gray-400 uppercase tracking-wider">Target Split Focus</p>
                <h4 className="text-xl font-extrabold text-white mt-1 uppercase">
                  {recommendation.focus_areas ? recommendation.focus_areas.join(" & ") : "Full Body Conditioning"}
                </h4>
              </div>
              <div className="flex space-x-4">
                <div className="bg-dark-bg px-4 py-2 rounded-xl border border-dark-border text-center">
                  <span className="text-[10px] text-gray-500 uppercase font-black">Difficulty</span>
                  <p className="text-sm font-bold text-brand-red mt-0.5 uppercase">{recommendation.difficulty || 'Intermediate'}</p>
                </div>
                <div className="bg-dark-bg px-4 py-2 rounded-xl border border-dark-border text-center">
                  <span className="text-[10px] text-gray-500 uppercase font-black">Est. Duration</span>
                  <p className="text-sm font-bold text-white mt-0.5">{recommendation.estimated_duration || 45} mins</p>
                </div>
              </div>
            </div>

            <div>
              <p className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">Recommended Exercises</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {recommendation.routine ? (
                  recommendation.routine.map((ex, idx) => (
                    <div key={idx} className="bg-dark-bg border border-dark-border p-4 rounded-xl flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="bg-brand-red/10 text-brand-red w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs">
                          {idx + 1}
                        </div>
                        <div>
                          <p className="text-sm font-bold text-white">{ex.exercise}</p>
                          <span className="text-[10px] text-gray-500 font-bold uppercase">{ex.muscle_group}</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-black text-white">{ex.sets} x {ex.reps}</p>
                        <span className="text-[10px] text-gray-500 font-bold uppercase">{ex.weight}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  recommendation.recommended_exercises && recommendation.recommended_exercises.map((ex, idx) => (
                    <div key={idx} className="bg-dark-bg border border-dark-border p-4 rounded-xl flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="bg-brand-red/10 text-brand-red w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs">
                          {idx + 1}
                        </div>
                        <div>
                          <p className="text-sm font-bold text-white">{ex.name}</p>
                          <span className="text-[10px] text-gray-500 font-bold uppercase">{ex.muscle}</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-black text-white">{ex.sets} x {ex.reps}</p>
                        <span className="text-[10px] text-gray-500 font-bold uppercase">{ex.weight || 'Bodyweight'}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="py-6 text-center text-gray-500 text-sm font-semibold">
            No recommendation routine available. Create your profile details to generate daily splits.
          </div>
        )}
      </div>
    </div>
  );
}
