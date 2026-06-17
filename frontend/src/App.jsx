import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

// Layout & Pages
import Layout from './components/Layout';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import Workouts from './pages/Workouts';
import WorkoutPage from './pages/WorkoutPage';
import CalorieTracker from './pages/CalorieTracker';
import OneRepMax from './pages/OneRepMax';
import CarbCycling from './pages/CarbCycling';
import FitnessChat from './pages/FitnessChat';
import BodyVision from './pages/BodyVision';
import ProfileSetup from './pages/ProfileSetup';

const API_BASE_URL = 'http://localhost:8000';
axios.defaults.withCredentials = true;

function App() {
  const [user, setUser] = useState(null);
  const [checkingAuth, setCheckingAuth] = useState(true);

  // Verify session authentication on load
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/api/user-status/`);
        if (response.data.authenticated) {
          setUser({ username: response.data.username });
        }
      } catch (err) {
        console.log("Not logged in");
      } finally {
        setCheckingAuth(false);
      }
    };
    checkAuthStatus();
  }, []);

  if (checkingAuth) {
    return (
      <div className="min-h-screen bg-dark-bg text-white flex flex-col items-center justify-center space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-400 text-sm font-semibold">Synchronizing with Stay Hard server...</p>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        {/* Unauthenticated Routes */}
        <Route 
          path="/login" 
          element={user ? <Navigate to="/" replace /> : <Login setUser={setUser} />} 
        />
        <Route 
          path="/signup" 
          element={user ? <Navigate to="/" replace /> : <Signup setUser={setUser} />} 
        />

        {/* Authenticated Routes wrapped in Sidebar Layout */}
        <Route
          path="/*"
          element={
            user ? (
              <Layout user={user} setUser={setUser}>
                <Routes>
                  <Route path="/" element={<Dashboard user={user} />} />
                  <Route path="/workouts" element={<Workouts />} />
                  <Route path="/workout/:workout_name" element={<WorkoutPage />} />
                  <Route path="/diet" element={<CalorieTracker />} />
                  <Route path="/1rm" element={<OneRepMax />} />
                  <Route path="/carb-cycling" element={<CarbCycling />} />
                  <Route path="/chat" element={<FitnessChat />} />
                  <Route path="/body-vision" element={<BodyVision />} />
                  <Route path="/profile" element={<ProfileSetup />} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Layout>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
