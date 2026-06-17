import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Send, RefreshCw, Sparkles, User, ShieldCheck, HelpCircle } from 'lucide-react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function FitnessChat() {
  const [messages, setMessages] = useState([]);
  const [welcomeMessage, setWelcomeMessage] = useState('');
  const [userSummary, setUserSummary] = useState('');
  
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [isGemini, setIsGemini] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchChatHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const fetchChatHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/chat/`);
      setMessages(res.data.messages || []);
      setWelcomeMessage(res.data.welcome_message || '');
      setUserSummary(res.data.user_summary || '');
      setIsGemini(res.data.is_gemini_active || false);
    } catch (err) {
      console.error("Failed to load chat history", err);
    } finally {
      setFetching(false);
    }
  };

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMsg = inputText.trim();
    setInputText('');
    setLoading(true);

    // Optimistic UI updates
    setMessages(prev => [...prev, { message: userMsg, response: '', timestamp: new Date().toISOString(), loading: true }]);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/chat/`, {
        message: userMsg
      });

      // Update message bubble with actual AI response
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (lastIdx >= 0) {
          updated[lastIdx] = {
            message: userMsg,
            response: res.data.response,
            timestamp: new Date().toISOString(),
            intent: res.data.intent,
            tier: res.data.tier
          };
        }
        return updated;
      });
    } catch (err) {
      console.error(err);
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (lastIdx >= 0) {
          updated[lastIdx] = {
            message: userMsg,
            response: 'Error: Failed to fetch coaching response. Check your local connection or API key constraints.',
            timestamp: new Date().toISOString(),
            isError: true
          };
        }
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = async () => {
    if (!window.confirm("Are you sure you want to clear your coaching conversation history?")) return;
    try {
      await axios.get(`${API_BASE_URL}/clear-chat/`); // Calls the redirect endpoint, but Django will clear the session
      setMessages([]);
      setWelcomeMessage("Welcome to OS Architect. I am your Senior Fitness & Nutrition Coach. Let's build your transformation protocol or address your biomechanics queries.");
    } catch (err) {
      console.error("Failed to clear chat session", err);
    }
  };

  if (fetching) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-400 text-sm font-semibold">Initiating AI Fitness Coach...</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-120px)]">
      
      {/* Left Sidebar: Profile & Agent Constraints */}
      <div className="hidden lg:flex lg:col-span-1 bg-dark-card border border-dark-border rounded-3xl p-6 flex-col justify-between overflow-y-auto shadow-xl">
        <div className="space-y-6">
          <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
            <User className="w-5 h-5 text-brand-red" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Your Trainer</h3>
          </div>

          <div className="space-y-4 text-xs">
            <div className="bg-dark-bg border border-dark-border p-4 rounded-2xl">
              <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider block mb-1">Target Profile</span>
              <p className="text-gray-300 leading-relaxed font-semibold italic">
                {userSummary || "Setup your biometric profile to generate detailed coach constraints."}
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center space-x-2 text-gray-400">
                <ShieldCheck className="w-4 h-4 text-brand-red shrink-0" />
                <span>Intent-based response classifier</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-400">
                <Sparkles className="w-4 h-4 text-brand-red shrink-0 animate-pulse" />
                <span>{isGemini ? "Gemini Pro 1.5 Cognitive Engine" : "Local Ollama Fallback Engine"}</span>
              </div>
            </div>
          </div>
        </div>

        <button
          onClick={handleClearChat}
          className="w-full flex items-center justify-center space-x-2 bg-dark-border hover:bg-brand-red/10 hover:text-brand-red text-gray-400 py-3 rounded-xl text-xs font-bold transition-all duration-200 cursor-pointer"
        >
          <RefreshCw className="w-4 h-4" />
          <span>CLEAR CHAT HISTORY</span>
        </button>
      </div>

      {/* Right Content Pane: Chat Area */}
      <div className="lg:col-span-3 bg-dark-card border border-dark-border rounded-3xl flex flex-col justify-between overflow-hidden shadow-xl">
        {/* Messages Header */}
        <div className="p-4 bg-dark-card border-b border-dark-border flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-2.5 h-2.5 bg-brand-red rounded-full animate-pulse"></div>
            <span className="text-xs font-bold uppercase tracking-wider text-white">Coach OS Architect</span>
          </div>
          <button
            onClick={handleClearChat}
            className="lg:hidden text-xs text-gray-400 hover:text-brand-red font-bold"
          >
            Clear
          </button>
        </div>

        {/* Message Bubble Stream */}
        <div className="flex-1 p-6 overflow-y-auto space-y-4">
          {/* Welcome Message */}
          {welcomeMessage && (
            <div className="flex items-start space-x-3 max-w-[85%]">
              <div className="w-8 h-8 rounded-lg bg-brand-red/10 text-brand-red flex items-center justify-center shrink-0 font-bold text-xs">
                AI
              </div>
              <div className="bg-dark-bg border border-dark-border p-4 rounded-2xl rounded-tl-none">
                <p className="text-xs text-gray-300 leading-relaxed">{welcomeMessage}</p>
              </div>
            </div>
          )}

          {/* User & AI Messages */}
          {messages.map((m, idx) => (
            <React.Fragment key={idx}>
              {/* User Bubble */}
              <div className="flex items-start space-x-3 justify-end">
                <div className="bg-brand-red text-white p-4 rounded-2xl rounded-tr-none text-xs max-w-[85%] font-medium">
                  {m.message}
                </div>
              </div>

              {/* Bot Bubble */}
              {(m.response || m.loading) && (
                <div className="flex items-start space-x-3 max-w-[85%]">
                  <div className="w-8 h-8 rounded-lg bg-brand-red/10 text-brand-red flex items-center justify-center shrink-0 font-bold text-xs">
                    AI
                  </div>
                  <div className={`p-4 rounded-2xl rounded-tl-none border text-xs leading-relaxed ${
                    m.isError 
                      ? 'bg-brand-red/10 border-brand-red/30 text-brand-red' 
                      : 'bg-dark-bg border-dark-border text-gray-300'
                  }`}>
                    {m.loading ? (
                      <div className="flex items-center space-x-2 py-1">
                        <span className="w-1.5 h-1.5 bg-brand-red rounded-full animate-bounce"></span>
                        <span className="w-1.5 h-1.5 bg-brand-red rounded-full animate-bounce [animation-delay:0.2s]"></span>
                        <span className="w-1.5 h-1.5 bg-brand-red rounded-full animate-bounce [animation-delay:0.4s]"></span>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="whitespace-pre-line">{m.response}</p>
                        {m.intent && (
                          <span className="inline-block text-[8px] bg-dark-border text-gray-500 font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">
                            Intent: {m.intent} • {m.tier}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </React.Fragment>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* Input Bar */}
        <form onSubmit={handleSendMessage} className="p-4 bg-dark-bg border-t border-dark-border flex space-x-3">
          <input
            type="text"
            required
            disabled={loading}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask your coach: 'How do I squat with lower back issues?' or 'Should I do low carb today?'"
            className="flex-1 bg-dark-card border border-dark-border rounded-xl px-5 py-3 text-xs focus:outline-none focus:border-brand-red transition-all duration-200"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-brand-red hover:bg-brand-red-hover text-white p-3.5 rounded-xl flex items-center justify-center transition-all duration-200 cursor-pointer disabled:opacity-50"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>

      </div>

    </div>
  );
}
