import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Flame, Calculator, Sparkles, Loader2, RefreshCw } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function CarbCycling() {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('male');
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [heightUnit, setHeightUnit] = useState('cm');
  const [weightUnit, setWeightUnit] = useState('kg');
  const [activity, setActivity] = useState('moderate');
  const [goal, setGoal] = useState('maintenance');
  const [trainingDays, setTrainingDays] = useState(4);

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    // Attempt to pre-fill from user profile
    const loadProfileData = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/profile-setup/`);
        if (response.data.has_profile) {
          const p = response.data;
          setAge(p.age || '');
          setGender(p.gender === 'F' ? 'female' : 'male');
          setHeight(p.height || '');
          setWeight(p.weight || '');
          setGoal(p.primary_goal || 'maintenance');
        }
      } catch (err) {
        console.error("Failed to pre-fill profile data", err);
      }
    };
    loadProfileData();
  }, []);

  const handleCalculate = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    setResults(null);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/carb-cycling/`, {
        age: parseInt(age),
        gender,
        height: parseFloat(height),
        weight: parseFloat(weight),
        height_unit: heightUnit,
        weight_unit: weightUnit,
        activity_level: activity,
        goal,
        training_days: parseInt(trainingDays)
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Calculation failed. Verify input parameters.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">Carb Cycling Planner</h2>
        <p className="text-gray-400 text-sm mt-1">Generate dynamic macronutrient protocols matching your training frequency and rest cycles.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Input Parameters Form */}
        <div className="lg:col-span-1 bg-dark-card border border-dark-border p-6 rounded-3xl h-fit shadow-lg space-y-4">
          <div className="flex items-center space-x-2 border-b border-dark-border pb-3 mb-2">
            <Calculator className="w-5 h-5 text-brand-red" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Input Parameters</h3>
          </div>

          {error && (
            <div className="bg-brand-red/10 border border-brand-red/50 text-brand-red text-xs p-3 rounded-lg text-center font-bold">
              {error}
            </div>
          )}

          <form onSubmit={handleCalculate} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Age</label>
                <input 
                  type="number"
                  required
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
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
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Height</label>
                <div className="flex">
                  <input 
                    type="number"
                    step="0.1"
                    required
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                    className="w-full bg-dark-bg border border-dark-border rounded-l-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
                  />
                  <select
                    value={heightUnit}
                    onChange={(e) => setHeightUnit(e.target.value)}
                    className="bg-dark-border border-y border-r border-dark-border text-gray-300 rounded-r-xl px-2 py-2 text-xs focus:outline-none"
                  >
                    <option value="cm">cm</option>
                    <option value="ft">ft</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Weight</label>
                <div className="flex">
                  <input 
                    type="number"
                    step="0.1"
                    required
                    value={weight}
                    onChange={(e) => setWeight(e.target.value)}
                    className="w-full bg-dark-bg border border-dark-border rounded-l-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
                  />
                  <select
                    value={weightUnit}
                    onChange={(e) => setWeightUnit(e.target.value)}
                    className="bg-dark-border border-y border-r border-dark-border text-gray-300 rounded-r-xl px-2 py-2 text-xs focus:outline-none"
                  >
                    <option value="kg">kg</option>
                    <option value="lbs">lbs</option>
                  </select>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Activity Level</label>
              <select
                value={activity}
                onChange={(e) => setActivity(e.target.value)}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              >
                <option value="sedentary">Sedentary (Office job)</option>
                <option value="light">Lightly Active (1-2 days/wk)</option>
                <option value="moderate">Moderately Active (3-5 days/wk)</option>
                <option value="active">Very Active (6-7 days/wk)</option>
                <option value="very_active">Athlete / Physical Labor</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Primary Target Goal</label>
              <select
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              >
                <option value="maintenance">Maintain Weight</option>
                <option value="weight_loss">Weight Loss (Deficit)</option>
                <option value="muscle_gain">Muscle Gain (Surplus)</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Weekly Training Days</label>
              <input 
                type="number"
                min="1"
                max="7"
                required
                value={trainingDays}
                onChange={(e) => setTrainingDays(e.target.value)}
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-brand-red hover:bg-brand-red-hover text-white py-3 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Calculating Macros...</span>
                </>
              ) : (
                <span>CALCULATE PROTOCOL</span>
              )}
            </button>
          </form>
        </div>

        {/* Results Presentation Grid */}
        <div className="lg:col-span-2">
          {results ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {/* Daily Energy Metrics Header */}
              <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg grid grid-cols-2 gap-4 text-center">
                <div className="p-4 bg-dark-bg border border-dark-border rounded-2xl">
                  <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest block">Basal Metabolic Rate (BMR)</span>
                  <p className="text-2xl font-black text-white mt-1 leading-none">{results.bmr} kcal</p>
                </div>
                <div className="p-4 bg-dark-bg border border-dark-border rounded-2xl">
                  <span className="text-[9px] text-gray-500 font-bold uppercase tracking-widest block">Total Daily Expenditure (TDEE)</span>
                  <p className="text-2xl font-black text-brand-red mt-1 leading-none">{results.tdee} kcal</p>
                </div>
              </div>

              {/* Side-by-side Day protocols */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* High Carb Card */}
                <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg relative overflow-hidden flex flex-col justify-between">
                  <div className="absolute top-4 right-4 bg-brand-red/10 text-brand-red px-2.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider">
                    Training Day
                  </div>
                  
                  <div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block">HIGH CARB DAY</span>
                    <h3 className="text-3xl font-black text-white mt-2 leading-none">
                      {results.high_carb_day.calories} <span className="text-xs text-gray-400 font-bold">kcal</span>
                    </h3>

                    {/* Macro Breakdown */}
                    <div className="space-y-3 mt-6">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-semibold">Carbohydrates</span>
                        <span className="text-white font-bold">{results.high_carb_day.carbs_g}g</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-semibold">Protein</span>
                        <span className="text-white font-bold">{results.high_carb_day.protein_g}g</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-semibold">Fats</span>
                        <span className="text-white font-bold">{results.high_carb_day.fat_g}g</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Low Carb Card */}
                <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg relative overflow-hidden flex flex-col justify-between">
                  <div className="absolute top-4 right-4 bg-yellow-500/10 text-yellow-500 px-2.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider">
                    Rest Day
                  </div>

                  <div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block">LOW CARB DAY</span>
                    <h3 className="text-3xl font-black text-white mt-2 leading-none">
                      {results.low_carb_day.calories} <span className="text-xs text-gray-400 font-bold">kcal</span>
                    </h3>

                    {/* Macro Breakdown */}
                    <div className="space-y-3 mt-6">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-semibold">Carbohydrates</span>
                        <span className="text-white font-bold">{results.low_carb_day.carbs_g}g</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-semibold">Protein</span>
                        <span className="text-white font-bold">{results.low_carb_day.protein_g}g</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 font-semibold">Fats</span>
                        <span className="text-white font-bold">{results.low_carb_day.fat_g}g</span>
                      </div>
                    </div>
                  </div>
                </div>

              </div>

            </motion.div>
          ) : (
            <div className="bg-dark-card border border-dark-border border-dashed p-12 rounded-3xl text-center text-gray-500 text-sm font-semibold h-full flex flex-col justify-center items-center space-y-3">
              <Flame className="w-8 h-8 text-gray-600 animate-pulse" />
              <p>Generate metabolic calculations to formulate weekly macro cycling split targets.</p>
            </div>
          )}
        </div>

      </div>
    </motion.div>
  );
}
