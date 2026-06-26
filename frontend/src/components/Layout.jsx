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
  Flame 
} from 'lucide-react';
import axios from 'axios';

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
    <div className="min-h-screen bg-dark-bg text-slate-900 flex flex-col md:flex-row">
      {/* Sidebar Navigation */}
      <aside className="w-full md:w-64 bg-dark-card border-r border-dark-border flex flex-col justify-between shrink-0 shadow-sm">
        <div>
          {/* Brand Header */}
          <div className="p-6 border-b border-dark-border flex items-center space-x-3">
            <div className="bg-brand-red p-2 rounded-xl flex items-center justify-center shadow-md shadow-brand-red/10">
              <Dumbbell className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-extrabold tracking-tight text-slate-900 m-0 leading-none">STAY HARD</h1>
              <span className="text-[10px] uppercase tracking-widest text-slate-400 font-bold mt-1 block">Fitness Analyser</span>
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
                  className={`flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-bold transition-all duration-200 ${
                    isActive 
                      ? 'bg-brand-red text-white shadow-lg shadow-brand-red/15' 
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
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
          <div className="p-4 border-t border-dark-border bg-slate-50/50">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-brand-red font-black text-sm">
                  {user.username.substring(0, 2).toUpperCase()}
                </div>
                <div className="truncate max-w-[120px]">
                  <p className="text-sm font-black truncate text-slate-900 leading-none m-0">{user.username}</p>
                  <span className="text-[10px] text-slate-400 font-bold block mt-1">Athlete</span>
                </div>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center justify-center space-x-2 bg-slate-100 hover:bg-brand-red/10 hover:text-brand-red text-slate-600 py-2.5 rounded-xl text-xs font-bold transition-all duration-200 cursor-pointer"
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
