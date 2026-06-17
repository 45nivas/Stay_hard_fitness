import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Calculator, Zap, Percent, Loader2 } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function OneRepMax() {
  const [weight, setWeight] = useState('');
  const [reps, setReps] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleCalculate = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    setResults(null);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/one-rep-max/`, {
        weight: parseFloat(weight),
        reps: parseInt(reps)
      });
      setResults(res.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Calculation failed. Check your values.');
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
        <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">One-Rep Max (1RM) Calculator</h2>
        <p className="text-gray-400 text-sm mt-1">Estimate your maximum lift capacity and calculate training intensity splits.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Form Card */}
        <div className="lg:col-span-1 bg-dark-card border border-dark-border p-6 rounded-3xl h-fit shadow-lg">
          <div className="flex items-center space-x-2 border-b border-dark-border pb-3 mb-5">
            <Calculator className="w-5 h-5 text-brand-red" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Calculator Input</h3>
          </div>

          {error && (
            <div className="bg-brand-red/10 border border-brand-red/50 text-brand-red text-xs p-3 rounded-lg mb-4 text-center font-bold">
              {error}
            </div>
          )}

          <form onSubmit={handleCalculate} className="space-y-4">
            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Weight Lifted (kg / lbs)</label>
              <input 
                type="number"
                step="0.1"
                required
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
                placeholder="e.g., 100"
                className="w-full bg-dark-bg border border-dark-border rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-brand-red transition-all duration-200"
              />
            </div>

            <div>
              <label className="block text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1.5">Repetitions Completed</label>
              <input 
                type="number"
                required
                value={reps}
                onChange={(e) => setReps(e.target.value)}
                placeholder="e.g., 5"
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
                  <span>Calculating...</span>
                </>
              ) : (
                <span>CALCULATE 1RM</span>
              )}
            </button>
          </form>
        </div>

        {/* Results Card */}
        <div className="lg:col-span-2">
          {results ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {/* Highlight Max Output */}
              <div className="bg-dark-card border border-dark-border p-6 rounded-3xl flex flex-col md:flex-row justify-between items-center md:items-stretch gap-6 shadow-lg">
                <div className="text-center md:text-left flex flex-col justify-between">
                  <div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Calculated One-Rep Max</span>
                    <h3 className="text-5xl font-black text-brand-red leading-none mt-2 select-none">
                      {results.one_rep_max} <span className="text-lg text-gray-400 font-bold">kg/lbs</span>
                    </h3>
                  </div>
                  <p className="text-xs text-gray-400 mt-4 leading-relaxed max-w-sm">
                    Calculated by averaging the **Epley**, **Brzycki**, and **Lander** formulas to reduce individual variance.
                  </p>
                </div>

                <div className="bg-dark-bg border border-dark-border p-4 rounded-2xl flex-1 grid grid-cols-3 gap-4 text-center items-center">
                  <div>
                    <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider block">Epley</span>
                    <p className="text-sm font-bold text-white mt-1">{results.epley}</p>
                  </div>
                  <div className="border-x border-dark-border">
                    <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider block">Brzycki</span>
                    <p className="text-sm font-bold text-white mt-1">{results.brzycki}</p>
                  </div>
                  <div>
                    <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider block">Lander</span>
                    <p className="text-sm font-bold text-white mt-1">{results.lander}</p>
                  </div>
                </div>
              </div>

              {/* Training Percentage Split Grid */}
              <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg">
                <div className="flex items-center space-x-2 border-b border-dark-border pb-3 mb-4">
                  <Percent className="w-5 h-5 text-brand-red" />
                  <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Intensity Splits (Workload Matrix)</h3>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                  {Object.entries(results.percentages).map(([pct, val]) => (
                    <div key={pct} className="bg-dark-bg border border-dark-border p-3.5 rounded-xl text-center">
                      <span className="text-[10px] text-brand-red font-black uppercase tracking-wider">{pct}</span>
                      <p className="text-base font-black text-white mt-1 leading-none">{val}</p>
                      <span className="text-[8px] text-gray-500 font-bold uppercase block mt-1">
                        {parseInt(pct) >= 85 ? 'Strength' : parseInt(pct) >= 70 ? 'Hypertrophy' : 'Endurance'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

            </motion.div>
          ) : (
            <div className="bg-dark-card border border-dark-border border-dashed p-12 rounded-3xl text-center text-gray-500 text-sm font-semibold h-full flex flex-col justify-center items-center space-y-3">
              <Zap className="w-8 h-8 text-gray-600 animate-bounce" />
              <p>Perform a calculation to display maximum lifts and training splits.</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
