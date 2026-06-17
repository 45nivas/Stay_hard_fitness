import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Dumbbell, Play, Award, Zap } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function Workouts() {
  const [workouts, setWorkouts] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchWorkouts = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/workouts/`);
        setWorkouts(res.data.workouts);
      } catch (err) {
        console.error("Error fetching workouts", err);
      } finally {
        setLoading(false);
      }
    };
    fetchWorkouts();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-400 text-sm font-semibold">Loading exercise protocols...</p>
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
        <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">AI Pose Correction</h2>
        <p className="text-gray-400 text-sm mt-1">Activate your web camera to track reps and calibrate your biomechanics in real-time.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {workouts.map((w, idx) => (
          <motion.div
            key={w.slug}
            whileHover={{ scale: 1.02 }}
            className="bg-dark-card border border-dark-border p-6 rounded-2xl flex flex-col justify-between hover:border-dark-border-hover transition-all duration-300"
          >
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-[10px] font-bold text-brand-red bg-brand-red/10 px-2.5 py-0.5 rounded-full uppercase tracking-wider">
                  {w.category}
                </span>
                <Dumbbell className="w-5 h-5 text-gray-500" />
              </div>
              <h3 className="text-xl font-extrabold text-white mb-2">{w.name}</h3>
              
              <div className="space-y-2 mt-4">
                <div className="flex items-center space-x-2 text-xs text-gray-400">
                  <Zap className="w-4 h-4 text-brand-red shrink-0" />
                  <span>Real-time repetition counter</span>
                </div>
                <div className="flex items-center space-x-2 text-xs text-gray-400">
                  <Award className="w-4 h-4 text-brand-red shrink-0" />
                  <span>Angle analysis & range-of-motion detection</span>
                </div>
              </div>
            </div>

            <button
              onClick={() => navigate(`/workout/${w.slug}`)}
              className="mt-8 w-full bg-brand-red hover:bg-brand-red-hover text-white py-3 rounded-xl font-bold text-sm flex items-center justify-center space-x-2 transition-all duration-200 cursor-pointer"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>START CORRECTION</span>
            </button>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
