import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Utensils, 
  Mic, 
  MicOff, 
  Plus, 
  Trash2, 
  Loader2, 
  Sparkles, 
  ChevronRight, 
  Calculator 
} from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function CalorieTracker() {
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  
  const [summary, setSummary] = useState({
    total_calories: 0,
    total_protein: 0,
    total_carbs: 0,
    total_fat: 0,
    target_calories: 2000,
    target_protein: 150,
    target_carbs: 220,
    target_fat: 65
  });
  const [loggedMeals, setLoggedMeals] = useState([]);
  
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState('');

  // Voice logging parser candidates states
  const [candidates, setCandidates] = useState([]);
  const [foodQuery, setFoodQuery] = useState('');
  const [showPicker, setShowPicker] = useState(false);

  // Native Browser Speech Recognition Setup
  const recognitionRef = useRef(null);

  useEffect(() => {
    fetchDailySummary();

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = false;
      rec.interimResults = false;
      rec.lang = 'en-US';

      rec.onstart = () => setIsRecording(true);
      rec.onend = () => setIsRecording(false);
      rec.onerror = (e) => {
        console.error("Speech recognition error", e);
        setIsRecording(false);
      };
      rec.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setInputText(transcript);
      };
      recognitionRef.current = rec;
    }
  }, []);

  const fetchDailySummary = async () => {
    try {
      const summaryRes = await axios.get(`${API_BASE_URL}/api/get-daily-summary/`);
      const mealsRes = await axios.get(`${API_BASE_URL}/api/get-daily-meals/`);
      
      setSummary(summaryRes.data);
      setLoggedMeals(mealsRes.data.meals || []);
    } catch (err) {
      console.error("Error loading daily calories summary", err);
    } finally {
      setFetching(false);
    }
  };

  const handleToggleRecord = () => {
    if (!recognitionRef.current) {
      alert("Speech Recognition is not supported in this browser. Try Google Chrome.");
      return;
    }
    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      setInputText('');
      recognitionRef.current.start();
    }
  };

  const handleVoiceLogSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    setError('');
    setLoading(true);
    setCandidates([]);
    setShowPicker(false);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/voice-log/`, {
        text: inputText
      });

      const data = res.data;

      if (data.mode === 'auto') {
        // Automatically logged using preference
        setInputText('');
        fetchDailySummary();
      } else if (data.mode === 'picker') {
        // Ask user to pick a candidate
        setCandidates(data.candidates);
        setFoodQuery(data.food_query);
        setShowPicker(true);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to parse meal query. Try typing something simpler.');
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmCandidate = async (candidate) => {
    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/api/confirm-meal/`, {
        food_query: foodQuery,
        chosen_food_name: candidate.food,
        chosen_food_data: {
          calories: candidate.calories,
          protein: candidate.protein,
          carbs: candidate.carbs,
          fat: candidate.fat,
          fiber: candidate.fiber || 0,
          sodium: candidate.sodium || 0
        },
        quantity: candidate.quantity || 100,
        unit: candidate.unit || 'g'
      });

      // Reset
      setShowPicker(false);
      setCandidates([]);
      setInputText('');
      fetchDailySummary();
    } catch (err) {
      console.error("Confirm failed", err);
      setError("Failed to confirm food item.");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteMeal = async (mealId) => {
    try {
      const res = await axios.post(`${API_BASE_URL}/api/delete-meal/`, { id: mealId });
      if (res.data.success) {
        // Animated deletion
        setLoggedMeals(meals => meals.filter(m => m.id !== mealId));
        fetchDailySummary();
      }
    } catch (err) {
      console.error("Failed to delete meal log", err);
    }
  };

  const calculatePct = (val, max) => {
    if (!max) return 0;
    return Math.min(Math.round((val / max) * 100), 100);
  };

  if (fetching) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-400 text-sm font-semibold">Gathering daily nutrition data...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">Calorie & Macro Tracker</h2>
        <p className="text-gray-400 text-sm mt-1">Track your calories and nutrient partitions using natural voice dictation.</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Calorie Card */}
        <div className="lg:col-span-1 bg-dark-card border border-dark-border p-6 rounded-3xl flex flex-col justify-between shadow-lg">
          <div>
            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block mb-1">Calories Consumed</span>
            <div className="flex items-baseline space-x-1.5">
              <span className="text-4xl font-black text-white">{summary.total_calories}</span>
              <span className="text-xs text-gray-400 font-bold">/ {summary.target_calories || 2000} kcal</span>
            </div>
          </div>
          {/* Progress Bar */}
          <div className="w-full bg-dark-bg h-2.5 rounded-full overflow-hidden border border-dark-border mt-6">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${calculatePct(summary.total_calories, summary.target_calories)}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              className="bg-brand-red h-full rounded-full"
            />
          </div>
          <div className="flex justify-between items-center text-[10px] text-gray-500 font-bold uppercase tracking-wider mt-2">
            <span>{calculatePct(summary.total_calories, summary.target_calories)}% Complete</span>
            <span>{Math.max((summary.target_calories || 2000) - summary.total_calories, 0)} kcal left</span>
          </div>
        </div>

        {/* Macros Card */}
        <div className="lg:col-span-3 bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg space-y-6">
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block">Macronutrient Partitions</span>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Protein */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-white">Protein</span>
                <span className="text-gray-400 font-bold">{summary.total_protein}g / {summary.target_protein || 150}g</span>
              </div>
              <div className="w-full bg-dark-bg h-2 rounded-full overflow-hidden border border-dark-border">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${calculatePct(summary.total_protein, summary.target_protein)}%` }}
                  transition={{ duration: 0.5 }}
                  className="bg-brand-red h-full rounded-full"
                />
              </div>
              <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider block">
                {calculatePct(summary.total_protein, summary.target_protein)}% of target
              </span>
            </div>

            {/* Carbs */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-white">Carbohydrates</span>
                <span className="text-gray-400 font-bold">{summary.total_carbs}g / {summary.target_carbs || 220}g</span>
              </div>
              <div className="w-full bg-dark-bg h-2 rounded-full overflow-hidden border border-dark-border">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${calculatePct(summary.total_carbs, summary.target_carbs)}%` }}
                  transition={{ duration: 0.5 }}
                  className="bg-yellow-500 h-full rounded-full"
                />
              </div>
              <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider block">
                {calculatePct(summary.total_carbs, summary.target_carbs)}% of target
              </span>
            </div>

            {/* Fats */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="font-bold text-white">Fats</span>
                <span className="text-gray-400 font-bold">{summary.total_fat}g / {summary.target_fat || 65}g</span>
              </div>
              <div className="w-full bg-dark-bg h-2 rounded-full overflow-hidden border border-dark-border">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${calculatePct(summary.total_fat, summary.target_fat)}%` }}
                  transition={{ duration: 0.5 }}
                  className="bg-orange-500 h-full rounded-full"
                />
              </div>
              <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider block">
                {calculatePct(summary.total_fat, summary.target_fat)}% of target
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Action Block: Voice Log Input */}
      <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg relative overflow-hidden">
        <div className="flex items-center space-x-2 mb-4">
          <Sparkles className="w-5 h-5 text-brand-red animate-pulse" />
          <h3 className="text-lg font-bold text-white m-0 uppercase tracking-wider">Voice Macro Logger</h3>
        </div>

        {error && (
          <div className="bg-brand-red/10 border border-brand-red/50 text-brand-red text-xs p-3 rounded-lg mb-4 text-center font-bold">
            {error}
          </div>
        )}

        <form onSubmit={handleVoiceLogSubmit} className="space-y-4">
          <div className="relative">
            <textarea
              rows="3"
              required
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Speak or type: 'For lunch I ate 200g of grilled chicken, a cup of brown rice, and an avocado...'"
              className="w-full bg-dark-bg border border-dark-border rounded-2xl pl-5 pr-14 py-4 text-sm focus:outline-none focus:border-brand-red transition-all duration-200 resize-none leading-relaxed"
            />
            {/* Record button */}
            <button
              type="button"
              onClick={handleToggleRecord}
              className={`absolute right-4 top-4 p-3 rounded-xl flex items-center justify-center transition-all duration-200 cursor-pointer ${
                isRecording 
                  ? 'bg-brand-red text-white animate-pulse' 
                  : 'bg-dark-border text-gray-400 hover:text-white hover:bg-dark-border-hover'
              }`}
            >
              {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="bg-brand-red hover:bg-brand-red-hover text-white px-8 py-3.5 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer flex items-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Parsing Nutrition...</span>
                </>
              ) : (
                <span>LOG MEAL</span>
              )}
            </button>
          </div>
        </form>

        {/* Candidate Picker Modal Overlay */}
        <AnimatePresence>
          {showPicker && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6"
            >
              <motion.div 
                initial={{ scale: 0.95 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.95 }}
                className="bg-dark-card border border-dark-border p-6 rounded-2xl w-full max-w-lg shadow-2xl space-y-4 max-h-[90%] overflow-y-auto"
              >
                <div className="flex items-center space-x-2 pb-3 border-b border-dark-border">
                  <Calculator className="w-5 h-5 text-brand-red" />
                  <h4 className="text-sm font-bold text-white uppercase tracking-wider m-0">
                    Confirm Food Data: "{foodQuery}"
                  </h4>
                </div>

                <p className="text-xs text-gray-400">
                  Multiple variations detected. Select the matching item to save to your daily telemetry:
                </p>

                <div className="space-y-3">
                  {candidates.map((cand, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleConfirmCandidate(cand)}
                      className="w-full text-left bg-dark-bg border border-dark-border hover:border-brand-red p-4 rounded-xl flex justify-between items-center transition-all duration-200 group cursor-pointer"
                    >
                      <div>
                        <p className="text-sm font-bold text-white group-hover:text-brand-red transition-colors duration-200">
                          {cand.food}
                        </p>
                        <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">
                          {cand.quantity}{cand.unit} • P: {cand.protein}g • C: {cand.carbs}g • F: {cand.fat}g
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-black text-white">{cand.calories} kcal</span>
                        <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-brand-red transition-all duration-200" />
                      </div>
                    </button>
                  ))}
                </div>

                <div className="flex justify-end pt-2">
                  <button
                    onClick={() => setShowPicker(false)}
                    className="text-xs font-bold text-gray-500 hover:text-white transition-colors duration-200 cursor-pointer"
                  >
                    Cancel
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Daily Logs Table */}
      <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg">
        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block mb-4">Daily Food Ledger</span>
        
        <div className="space-y-3">
          <AnimatePresence initial={false}>
            {loggedMeals.length > 0 ? (
              loggedMeals.map((meal) => (
                <motion.div
                  key={meal.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                  className="bg-dark-bg border border-dark-border p-4 rounded-xl flex items-center justify-between group"
                >
                  <div className="flex items-center space-x-3">
                    <div className="bg-brand-red/10 text-brand-red w-8 h-8 rounded-lg flex items-center justify-center">
                      <Utensils className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-white uppercase">{meal.food}</p>
                      <span className="text-[10px] text-gray-500 font-bold uppercase">
                        {meal.quantity}{meal.unit} • P: {meal.protein}g • C: {meal.carbs}g • F: {meal.fat}g
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm font-black text-white">{meal.calories} kcal</span>
                    <button
                      onClick={() => handleDeleteMeal(meal.id)}
                      className="text-gray-500 hover:text-brand-red p-1 rounded transition-colors duration-200 opacity-0 group-hover:opacity-100 cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="text-center py-8 text-xs text-gray-500 font-semibold">
                No food logged today. Dictate or type your meals to get started!
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
