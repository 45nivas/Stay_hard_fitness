import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { 
  Dumbbell, 
  Utensils, 
  Calculator, 
  MessageSquare, 
  Camera, 
  BarChart2, 
  LogOut, 
  User, 
  Flame 
} from 'lucide-react';
import axios from 'axios';

// Backend base URL (change to empty string for production/Docker build since it will be served from the same host)
const API_BASE_URL = 'http://localhost:8000';
axios.defaults.withCredentials = true;

export default function Layout({ children, user, setUser }) {
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/logout/`);
      setUser(null);
      navigate('/login');
    } catch (err) {
      console.error("Logout failed", err);
    }
  };

  const navItems = [
    { name: 'Dashboard', path: '/', icon: BarChart2 },
    { name: 'Workouts', path: '/workouts', icon: Dumbbell },
    { name: 'Calorie Tracker', path: '/diet', icon: Utensils },
    { name: '1RM Calculator', path: '/1rm', icon: Calculator },
    { name: 'Carb Cycling', path: '/carb-cycling', icon: Flame },
    { name: 'AI Coach Chat', path: '/chat', icon: MessageSquare },
    { name: 'Body Vision', path: '/body-vision', icon: Camera },
  ];

  return (
    <div className="min-h-screen bg-dark-bg text-white flex flex-col md:flex-row">
      {/* Sidebar Navigation */}
      <aside className="w-full md:w-64 bg-dark-card border-r border-dark-border flex flex-col justify-between shrink-0">
        <div>
          {/* Brand Header */}
          <div className="p-6 border-b border-dark-border flex items-center space-x-3">
            <div className="bg-brand-red p-2 rounded-lg flex items-center justify-center">
              <Dumbbell className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white m-0 leading-none">STAY HARD</h1>
              <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Fitness Analyser</span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-4 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                    isActive 
                      ? 'bg-brand-red text-white shadow-lg shadow-brand-red/10' 
                      : 'text-gray-400 hover:text-white hover:bg-dark-border'
                  }`}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* User Profile Summary & Logout */}
        {user && (
          <div className="p-4 border-t border-dark-border bg-dark-bg/30">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-full bg-dark-border flex items-center justify-center text-brand-red font-bold text-sm">
                  {user.username.substring(0, 2).toUpperCase()}
                </div>
                <div className="truncate max-w-[120px]">
                  <p className="text-sm font-bold truncate text-white leading-none m-0">{user.username}</p>
                  <span className="text-[10px] text-gray-500">Athlete</span>
                </div>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center space-x-2 bg-dark-border hover:bg-brand-red/10 hover:text-brand-red text-gray-400 py-2.5 rounded-lg text-xs font-bold transition-all duration-200"
            >
              <LogOut className="w-4 h-4" />
              <span>Log Out</span>
            </button>
          </div>
        )}
      </aside>

      {/* Main Content Pane */}
      <main className="flex-1 bg-dark-bg p-6 md:p-10 overflow-y-auto max-h-screen">
        <div className="max-w-6xl mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}
