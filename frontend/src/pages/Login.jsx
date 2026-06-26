import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Dumbbell, Eye, EyeOff, Loader2 } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function Login({ setUser }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/login/`, {
        username,
        password
      });
      
      if (response.data.success) {
        setUser({ username: response.data.username });
        if (response.data.has_profile) {
          navigate('/');
        } else {
          navigate('/profile');
        }
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to authenticate. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex items-center justify-center p-6">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md bg-white border border-slate-200 p-8 rounded-3xl shadow-xl shadow-slate-100"
      >
        {/* Header Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="bg-brand-red p-3 rounded-2xl mb-3 flex items-center justify-center shadow-lg shadow-brand-red/15">
            <Dumbbell className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-2xl font-extrabold tracking-tight text-slate-900 m-0">Welcome Back</h2>
          <p className="text-slate-500 text-xs mt-1.5 font-medium">Sign in to your Stay Hard Fitness profile</p>
        </div>

        {error && (
          <div className="bg-brand-red/10 border border-brand-red/30 text-brand-red text-xs p-3 rounded-xl mb-6 text-center font-bold">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Username Input */}
          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Username</label>
            <input 
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-brand-red focus:bg-white transition-all duration-200"
            />
          </div>

          {/* Password Input */}
          <div>
            <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Password</label>
            <div className="relative">
              <input 
                type={showPassword ? "text" : "password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full bg-slate-50 border border-slate-200 rounded-xl pl-4 pr-12 py-3 text-sm focus:outline-none focus:border-brand-red focus:bg-white transition-all duration-200"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-900 transition-colors duration-200"
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-brand-red hover:bg-brand-red-hover text-white py-3.5 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer flex items-center justify-center space-x-2 shadow-lg shadow-brand-red/10"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Signing In...</span>
              </>
            ) : (
              <span>SIGN IN</span>
            )}
          </button>
        </form>

        {/* Signup Link */}
        <div className="mt-8 text-center text-xs text-slate-500 font-medium">
          New to Stay Hard Fitness?{' '}
          <Link to="/signup" className="text-brand-red hover:text-brand-red-hover font-bold hover:underline transition-all duration-200">
            Create an Account
          </Link>
        </div>
      </motion.div>
    </div>
  );
}
