import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { User, Target, Activity, ShieldAlert, Loader2 } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function ProfileSetup() {
  const [age, setAge] = useState('');
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [gender, setGender] = useState('M');
  const [fitnessLevel, setFitnessLevel] = useState('intermediate');
  const [primaryGoal, setPrimaryGoal] = useState('muscle_gain');
  const [injuries, setInjuries] = useState('');
  const [availableTime, setAvailableTime] = useState(60);
  const [weakMuscles, setWeakMuscles] = useState('');
  const [equipment, setEquipment] = useState('');
  const [calories, setCalories] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/profile-setup/`);
        if (response.data.has_profile) {
          const p = response.data;
          setAge(p.age || '');
          setHeight(p.height || '');
          setWeight(p.weight || '');
          setGender(p.gender || 'M');
          setFitnessLevel(p.fitness_level || 'intermediate');
          setPrimaryGoal(p.primary_goal || 'muscle_gain');
          setInjuries(p.injuries_or_limitations || '');
          setAvailableTime(p.available_time || 60);
          setWeakMuscles(p.weak_muscles || '');
          setEquipment(p.equipment_available || '');
          setCalories(p.calories_per_day || '');
        }
      } catch (err) {
        console.error("Failed to load profile", err);
      } finally {
        setFetching(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const payload = {
      age: parseInt(age),
      height: parseFloat(height),
      weight: parseFloat(weight),
      gender,
      fitness_level: fitnessLevel,
      primary_goal: primaryGoal,
      injuries_or_limitations: injuries,
      available_time: parseInt(availableTime),
      weak_muscles: weakMuscles,
      equipment_available: equipment,
      calories_per_day: calories ? parseInt(calories) : null
    };

    try {
      const res = await axios.post(`${API_BASE_URL}/api/profile-setup/`, payload);
      if (res.data.success) {
        navigate('/');
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to save profile. Please verify your inputs.');
    } finally {
      setLoading(false);
    }
  };

  if (fetching) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-400 text-sm font-semibold">Retrieving your athlete profile...</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">Setup Athlete Profile</h2>
        <p className="text-gray-400 text-sm mt-1">Configure your biometrics and parameters to initialize adaptive recommendations.</p>
      </div>

      {error && (
        <div className="bg-brand-red/10 border border-brand-red/50 text-brand-red text-xs p-3 rounded-lg text-center font-bold">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Section 1: Core Biometrics */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-2xl space-y-5">
            <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
              <User className="w-5 h-5 text-brand-red" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Biometric Details</h3>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Age</label>
                <input 
                  type="number"
                  required
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  placeholder="e.g., 25"
                  className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
                />
              </div>
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Gender</label>
                <select
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
                >
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="O">Other</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Height (cm)</label>
                <input 
                  type="number"
                  step="0.1"
                  required
                  value={height}
                  onChange={(e) => setHeight(e.target.value)}
                  placeholder="e.g., 180"
                  className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
                />
              </div>
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Weight (kg)</label>
                <input 
                  type="number"
                  step="0.1"
                  required
                  value={weight}
                  onChange={(e) => setWeight(e.target.value)}
                  placeholder="e.g., 82.5"
                  className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
                />
              </div>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Daily Calorie Target (Optional)</label>
              <input 
                type="number"
                value={calories}
                onChange={(e) => setCalories(e.target.value)}
                placeholder="Leave blank for automatic calculation"
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              />
            </div>
          </div>

          {/* Section 2: Fitness Goals & Settings */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-2xl space-y-5">
            <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
              <Target className="w-5 h-5 text-brand-red" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Goals & Level</h3>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Fitness Level</label>
              <select
                value={fitnessLevel}
                onChange={(e) => setFitnessLevel(e.target.value)}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              >
                <option value="beginner">Beginner (New to lifting)</option>
                <option value="intermediate">Intermediate (1-3 years experience)</option>
                <option value="advanced">Advanced (3+ years experience)</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Primary Goal</label>
              <select
                value={primaryGoal}
                onChange={(e) => setPrimaryGoal(e.target.value)}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              >
                <option value="muscle_gain">Muscle Gain (Hypertrophy)</option>
                <option value="strength">Build Strength (Powerlifting)</option>
                <option value="weight_loss">Weight Loss (Fat Loss)</option>
                <option value="endurance">Improve Endurance</option>
                <option value="general_fitness">General Fitness / Health</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Available Time Per Session (minutes)</label>
              <input 
                type="number"
                required
                value={availableTime}
                onChange={(e) => setAvailableTime(e.target.value)}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              />
            </div>
          </div>

          {/* Section 3: Training Parameters & Equipment */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-2xl space-y-5">
            <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
              <Activity className="w-5 h-5 text-brand-red" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Equipment & Focus</h3>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Equipment Available (Comma-separated)</label>
              <textarea 
                rows="2"
                value={equipment}
                onChange={(e) => setEquipment(e.target.value)}
                placeholder="e.g., dumbbells, barbell, pull-up bar, resistance bands"
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200 resize-none"
              />
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Weak / Lagging Muscle Groups (Comma-separated)</label>
              <input 
                type="text"
                value={weakMuscles}
                onChange={(e) => setWeakMuscles(e.target.value)}
                placeholder="e.g., chest, biceps, calves"
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              />
            </div>
          </div>

          {/* Section 4: Injury / Safety */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-2xl space-y-5">
            <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
              <ShieldAlert className="w-5 h-5 text-brand-red" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Injuries & Safety</h3>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Physical Injuries or Limitations</label>
              <textarea 
                rows="4"
                value={injuries}
                onChange={(e) => setInjuries(e.target.value)}
                placeholder="e.g., lower back pain, bad left knee, shoulder impingement. AI coach will design modifications around this."
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200 resize-none"
              />
            </div>
          </div>

        </div>

        {/* Submit */}
        <div className="flex justify-end pt-4">
          <button
            type="submit"
            disabled={loading}
            className="bg-brand-red hover:bg-brand-red-hover text-white px-8 py-3.5 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer flex items-center space-x-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Saving Profile...</span>
              </>
            ) : (
              <span>SAVE PROTOCOL</span>
            )}
          </button>
        </div>
      </form>
    </motion.div>
  );
}
