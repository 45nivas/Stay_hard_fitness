import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Dumbbell, 
  Play, 
  Award, 
  Zap, 
  Mic, 
  MicOff, 
  Trash2, 
  Loader2, 
  Sparkles, 
  Calendar,
  Flame
} from 'lucide-react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

const API_BASE_URL = 'http://localhost:8000';

const EXERCISE_LIST = [
  { id: 101, name: "Dumbbell Curl", muscle_group: "Biceps" },
  { id: 102, name: "Hammer Curl", muscle_group: "Biceps" },
  { id: 103, name: "Barbell Curl", muscle_group: "Biceps" },
  { id: 104, name: "Bench Press", muscle_group: "Chest" },
  { id: 105, name: "Incline Bench Press", muscle_group: "Chest" },
  { id: 106, name: "Dumbbell Bench Press", muscle_group: "Chest" },
  { id: 107, name: "Incline Dumbbell Press", muscle_group: "Chest" },
  { id: 108, name: "Shoulder Press", muscle_group: "Shoulders" },
  { id: 109, name: "Dumbbell Shoulder Press", muscle_group: "Shoulders" },
  { id: 110, name: "Lateral Raise", muscle_group: "Shoulders" },
  { id: 111, name: "Front Raise", muscle_group: "Shoulders" },
  { id: 112, name: "Rear Delt Fly", muscle_group: "Shoulders" },
  { id: 113, name: "Lat Pulldown", muscle_group: "Back" },
  { id: 114, name: "Seated Cable Row", muscle_group: "Back" },
  { id: 115, name: "Bent Over Row", muscle_group: "Back" },
  { id: 116, name: "Pull Up", muscle_group: "Back" },
  { id: 117, name: "Deadlift", muscle_group: "Back" },
  { id: 118, name: "Romanian Deadlift", muscle_group: "Legs" },
  { id: 119, name: "Squat", muscle_group: "Legs" },
  { id: 120, name: "Front Squat", muscle_group: "Legs" },
  { id: 121, name: "Leg Press", muscle_group: "Legs" },
  { id: 122, name: "Leg Extension", muscle_group: "Legs" },
  { id: 123, name: "Leg Curl", muscle_group: "Legs" },
  { id: 124, name: "Calf Raise", muscle_group: "Legs" },
  { id: 125, name: "Hip Thrust", muscle_group: "Legs" },
  { id: 126, name: "Chest Fly", muscle_group: "Chest" },
  { id: 127, name: "Tricep Pushdown", muscle_group: "Triceps" },
  { id: 128, name: "Overhead Tricep Extension", muscle_group: "Triceps" },
  { id: 129, name: "Dips", muscle_group: "Chest" }
];

export default function Workouts() {
  const [workouts, setWorkouts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [loggedWorkouts, setLoggedWorkouts] = useState([]);
  const [submittingLog, setSubmittingLog] = useState(false);
  const [error, setError] = useState('');
  
  // Voice engine & Whisper states
  const [voiceEngine, setVoiceEngine] = useState('native'); // 'native' or 'whisper'
  const [whisperStatus, setWhisperStatus] = useState('ready');
  const [previewLogs, setPreviewLogs] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const [toast, setToast] = useState({ visible: false, message: '', logs: [] });

  // PR progression modal states
  const [prModalOpen, setPrModalOpen] = useState(false);
  const [prModalExercise, setPrModalExercise] = useState('');
  const [prModalData, setPrModalData] = useState(null);
  const [prModalTab, setPrModalTab] = useState('e1rm');
  const [loadingPRData, setLoadingPRData] = useState(false);

  // Exercise picker state
  const [focusedExerciseIdx, setFocusedExerciseIdx] = useState(null);

  const handleOpenPRModal = async (exerciseIdOrName) => {
    let displayName = exerciseIdOrName;
    if (typeof exerciseIdOrName === 'number' || !isNaN(Number(exerciseIdOrName))) {
      const found = EXERCISE_LIST.find(e => e.id === Number(exerciseIdOrName));
      if (found) {
        displayName = found.name;
      }
    }
    setPrModalExercise(displayName);
    setPrModalOpen(true);
    setLoadingPRData(true);
    setPrModalData(null);
    try {
      const res = await axios.get(`${API_BASE_URL}/api/exercise-progress/${encodeURIComponent(exerciseIdOrName)}/`);
      if (res.data.success) {
        setPrModalData(res.data);
      } else {
        console.error("Failed to load PR progress data");
      }
    } catch (err) {
      console.error("Error loading PR progress data", err);
    } finally {
      setLoadingPRData(false);
    }
  };

  const navigate = useNavigate();
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    const fetchWorkoutsAndLogs = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/api/workouts/`);
        setWorkouts(res.data.workouts);
      } catch (err) {
        console.error("Error fetching workouts", err);
      } finally {
        setLoading(false);
      }

      fetchTodayWorkouts();
    };

    fetchWorkoutsAndLogs();

    // Native Speech Recognition Setup
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
        let transcript = event.results[0][0].transcript;
        // Clean common speech-to-text mishearings
        transcript = transcript
          .replace(/\blactose\b/gi, 'lat pulldown')
          .replace(/\blactoses\b/gi, 'lat pulldowns')
          .replace(/\blap\s+pulldown\b/gi, 'lat pulldown')
          .replace(/\blap\s+pulldowns\b/gi, 'lat pulldowns')
          .replace(/\blap\s+pull\s+down\b/gi, 'lat pulldown')
          .replace(/\blap\s+pull\s+downs\b/gi, 'lat pulldowns')
          .replace(/\bdid(?:n't)?\s+envelopes\b/gi, 'did incline bench press')
          .replace(/\benvelopes\b/gi, 'incline bench press')
          .replace(/\benvelope\b/gi, 'incline bench press')
          .replace(/\bany\b/gi, 'and')
          .replace(/\btensor\s+of\s+interest\s+in\s+the\s+/gi, 'bench press ')
          .replace(/\btensor\s+of\s+interest\b/gi, 'bench press')
          .replace(/\b5\s+day\s+or\s+30\s+pages\b/gi, '5 sets of 30 reps')
          .replace(/\byou\s+kind\s+of\s+will\s+transfer\b/gi, 'incline bench press')
          .replace(/\bskunk\s+crushes\b/gi, 'skull crushers')
          .replace(/\bskunk\s+crushers\b/gi, 'skull crushers')
          .replace(/\bone\b/gi, '1')
          .replace(/\btwo\b/gi, '2')
          .replace(/\bthree\b/gi, '3')
          .replace(/\bfour\b/gi, '4')
          .replace(/\bfive\b/gi, '5')
          .replace(/\bsix\b/gi, '6')
          .replace(/\bseven\b/gi, '7')
          .replace(/\beight\b/gi, '8')
          .replace(/\bnine\b/gi, '9')
          .replace(/\bten\b/gi, '10')
          .replace(/\beleven\b/gi, '11')
          .replace(/\btwelve\b/gi, '12');
        setInputText(transcript);
      };
      recognitionRef.current = rec;
    }

    // Cleanup audio tracks on unmount
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  // Auto-dismiss undo toast
  useEffect(() => {
    if (toast.visible) {
      const timer = setTimeout(() => {
        setToast({ visible: false, message: '', logs: [] });
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [toast.visible]);

  const fetchTodayWorkouts = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/api/get-workouts/`);
      setLoggedWorkouts(res.data.workouts || []);
    } catch (err) {
      console.error("Error fetching today's workouts", err);
    }
  };

  const handleStartWhisperRecord = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const mimeType = mediaRecorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        setIsRecording(false);
        
        try {
          setWhisperStatus('transcribing');
          
          const formData = new FormData();
          formData.append("audio", audioBlob, `audio.${mimeType.includes('mp4') ? 'mp4' : 'webm'}`);
          
          const res = await axios.post(`${API_BASE_URL}/api/transcribe-audio/`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          
          if (res.data.transcript) {
            let text = res.data.transcript;
            let cleanedText = text
              .replace(/\blactose\b/gi, 'lat pulldown')
              .replace(/\blactoses\b/gi, 'lat pulldowns')
              .replace(/\blap\s+pulldown\b/gi, 'lat pulldown')
              .replace(/\blap\s+pulldowns\b/gi, 'lat pulldowns')
              .replace(/\blap\s+pull\s+down\b/gi, 'lat pulldown')
              .replace(/\blap\s+pull\s+downs\b/gi, 'lat pulldowns')
              .replace(/\bdid(?:n't)?\s+envelopes\b/gi, 'did incline bench press')
              .replace(/\benvelopes\b/gi, 'incline bench press')
              .replace(/\benvelope\b/gi, 'incline bench press')
              .replace(/\bany\b/gi, 'and')
              .replace(/\btensor\s+of\s+interest\s+in\s+the\s+/gi, 'bench press ')
              .replace(/\btensor\s+of\s+interest\b/gi, 'bench press')
              .replace(/\b5\s+day\s+or\s+30\s+pages\b/gi, '5 sets of 30 reps')
              .replace(/\byou\s+kind\s+of\s+will\s+transfer\b/gi, 'incline bench press')
              .replace(/\bskunk\s+crushes\b/gi, 'skull crushers')
              .replace(/\bskunk\s+crushers\b/gi, 'skull crushers')
              .replace(/\bone\b/gi, '1')
              .replace(/\btwo\b/gi, '2')
              .replace(/\bthree\b/gi, '3')
              .replace(/\bfour\b/gi, '4')
              .replace(/\bfive\b/gi, '5')
              .replace(/\bsix\b/gi, '6')
              .replace(/\bseven\b/gi, '7')
              .replace(/\beight\b/gi, '8')
              .replace(/\bnine\b/gi, '9')
              .replace(/\bten\b/gi, '10')
              .replace(/\beleven\b/gi, '11')
              .replace(/\btwelve\b/gi, '12');
            setInputText(cleanedText);
            setWhisperStatus('ready');
          } else {
            setError("Could not transcribe audio. Please try again.");
            setWhisperStatus('ready');
          }
        } catch (err) {
          console.error("Audio transcription failed", err);
          setError("Failed to transcribe audio via Groq Whisper API.");
          setWhisperStatus('ready');
        }

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone access failed", err);
      setError("Microphone access denied or unavailable.");
    }
  };

  const handleStopWhisperRecord = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const handleToggleRecord = () => {
    setError('');
    if (voiceEngine === 'native') {
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
    } else {
      if (isRecording) {
        handleStopWhisperRecord();
      } else {
        setInputText('');
        handleStartWhisperRecord();
      }
    }
  };

  const handleVoiceLogSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    setError('');
    setSubmittingLog(true);

    const GYM_CORRECTIONS = {
      "envelopes": "incline bench press",
      "any": "and",
      "pulls": "pull ups",
      "pushes": "push ups",
      "dead lift": "deadlift",
      "roman": "Romanian deadlift",
      "shoulder press": "overhead press",
      "curls": "bicep curls",
      "extensions": "tricep extensions",
      "raises": "lateral raises"
    };

    let correctedText = inputText;
    for (const key of Object.keys(GYM_CORRECTIONS)) {
      const regex = new RegExp(key, 'gi');
      correctedText = correctedText.replace(regex, GYM_CORRECTIONS[key]);
    }

    try {
      const res = await axios.post(`${API_BASE_URL}/api/parse-workout-voice/`, {
        text: correctedText
      });
      if (res.data.success && res.data.parsed && res.data.parsed.length > 0) {
        setPreviewLogs(res.data.parsed);
        setShowPreview(true);
      } else {
        setError("Could not parse any exercises from text. Please try again.");
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to parse workout description.');
    } finally {
      setSubmittingLog(false);
    }
  };

  const handleConfirmLogs = async () => {
    try {
      const res = await axios.post(`${API_BASE_URL}/api/confirm-workout-log/`, {
        exercises: previewLogs
      });
      if (res.data.success && res.data.logged) {
        fetchTodayWorkouts();
        setShowPreview(false);
        setToast({
          visible: true,
          message: `Logged ${res.data.logged.length} exercise(s) successfully!`,
          logs: res.data.logged
        });
        setInputText('');
      }
    } catch (err) {
      console.error(err);
      alert(err.response?.data?.error || "Failed to confirm exercises.");
    }
  };

  const handleUndoLogs = async () => {
    if (!toast.logs || toast.logs.length === 0) return;
    try {
      await Promise.all(
        toast.logs.map(log => 
          axios.post(`${API_BASE_URL}/api/delete-workout/`, { id: log.id })
        )
      );
      fetchTodayWorkouts();
      setToast({ visible: false, message: '', logs: [] });
    } catch (err) {
      console.error("Failed to undo logging", err);
      alert("Failed to undo some workout logs.");
    }
  };

  const handleDeleteWorkout = async (logId) => {
    try {
      const res = await axios.post(`${API_BASE_URL}/api/delete-workout/`, { id: logId });
      if (res.data.success) {
        setLoggedWorkouts(workouts => workouts.filter(w => w.id !== logId));
      }
    } catch (err) {
      console.error("Failed to delete workout entry", err);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
        <div className="w-12 h-12 border-4 border-brand-red border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-500 text-sm font-semibold">Loading exercise protocols...</p>
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
        <h2 className="text-3xl font-black tracking-tight text-slate-900 m-0">Training & Biomechanics</h2>
        <p className="text-slate-500 text-sm mt-1.5 font-medium">Calibrate your posture using live computer vision, or log gym sets using natural voice dictation.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: AI Pose Correction selection */}
        <div className="lg:col-span-2 space-y-6">
          <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest block">Computer Vision Pose Correction</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {workouts.map((w) => (
              <motion.div
                key={w.slug}
                whileHover={{ scale: 1.01, y: -2 }}
                className="bg-dark-card border border-dark-border p-6 rounded-3xl flex flex-col justify-between hover:border-dark-border-hover shadow-sm transition-all duration-300"
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-[9px] font-black text-brand-red bg-brand-red/10 border border-brand-red/15 px-2.5 py-0.5 rounded-full uppercase tracking-widest">
                      {w.category}
                    </span>
                    <Dumbbell className="w-5 h-5 text-slate-400" />
                  </div>
                  <h3 className="text-lg font-black text-slate-900 mb-2">{w.name}</h3>
                  
                  <div className="space-y-2 mt-4">
                    <div className="flex items-center space-x-2 text-xs text-slate-500 font-semibold">
                      <Zap className="w-4 h-4 text-brand-red shrink-0" />
                      <span>Real-time repetition counter</span>
                    </div>
                    <div className="flex items-center space-x-2 text-xs text-slate-500 font-semibold">
                      <Award className="w-4 h-4 text-brand-red shrink-0" />
                      <span>Angle analysis & range-of-motion detection</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => navigate(`/workout/${w.slug}`)}
                  className="mt-8 w-full bg-brand-red hover:bg-brand-red-hover text-white py-3 rounded-xl font-bold text-sm flex items-center justify-center space-x-2 transition-all duration-200 cursor-pointer shadow-md shadow-brand-red/10"
                >
                  <Play className="w-4 h-4 fill-white text-white" />
                  <span>START CORRECTION</span>
                </button>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Right Column: Voice Workout Logger & History */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Voice Logger Card */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-sm relative overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Sparkles className="w-5 h-5 text-brand-red animate-pulse" />
                <h3 className="text-[10px] font-black text-slate-900 m-0 uppercase tracking-widest">Voice Workout Logger</h3>
              </div>
            </div>

            {/* Segmented Control */}
            <div className="flex bg-slate-100 p-1 rounded-2xl border border-slate-200/50 mb-4">
              <button
                type="button"
                onClick={() => setVoiceEngine('native')}
                className={`flex-1 text-center py-2 text-xs font-black rounded-xl transition-all duration-200 cursor-pointer ${
                  voiceEngine === 'native'
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-455 hover:text-slate-700'
                }`}
              >
                Browser Native
              </button>
              <button
                type="button"
                onClick={() => setVoiceEngine('whisper')}
                className={`flex-1 text-center py-2 text-xs font-black rounded-xl transition-all duration-200 flex items-center justify-center space-x-1.5 cursor-pointer ${
                  voiceEngine === 'whisper'
                    ? 'bg-brand-red text-white shadow-md shadow-brand-red/10'
                    : 'text-slate-455 hover:text-slate-700'
                }`}
              >
                <span>Whisper AI</span>
                <span className="text-[8px] bg-white/20 px-1.5 py-0.5 rounded-full uppercase tracking-wider">Cloud</span>
              </button>
            </div>

            {error && (
              <div className="bg-brand-red/10 border border-brand-red/30 text-brand-red text-xs p-3 rounded-xl mb-4 text-center font-bold">
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
                  disabled={whisperStatus === 'transcribing'}
                  placeholder={
                    whisperStatus === 'transcribing'
                      ? 'Whisper is transcribing your recorded audio...'
                      : "Speak or type: 'I did 3 sets of bench press for 12 reps at 80 kg and biceps curls...'"
                  }
                  className="w-full bg-slate-50 border border-slate-200 rounded-2xl pl-5 pr-14 py-4 text-sm focus:outline-none focus:border-brand-red focus:bg-white transition-all duration-200 resize-none leading-relaxed text-slate-800 font-medium shadow-inner disabled:bg-slate-100 disabled:text-slate-400"
                />
                
                {/* Soundwave animation for visual feedback */}
                {isRecording && (
                  <div className="absolute right-16 top-6 flex items-center space-x-1 h-5 pointer-events-none">
                    <span className="w-0.5 h-3 bg-brand-red rounded-full animate-bounce [animation-duration:0.6s]"></span>
                    <span className="w-0.5 h-5 bg-brand-red rounded-full animate-bounce [animation-duration:0.4s] [animation-delay:0.1s]"></span>
                    <span className="w-0.5 h-2 bg-brand-red rounded-full animate-bounce [animation-duration:0.8s] [animation-delay:0.2s]"></span>
                    <span className="w-0.5 h-4 bg-brand-red rounded-full animate-bounce [animation-duration:0.5s] [animation-delay:0.3s]"></span>
                    <span className="w-0.5 h-3 bg-brand-red rounded-full animate-bounce [animation-duration:0.7s] [animation-delay:0.15s]"></span>
                  </div>
                )}

                <button
                  type="button"
                  onClick={handleToggleRecord}
                  disabled={voiceEngine === 'whisper' && whisperStatus === 'loading'}
                  className={`absolute right-4 top-4 p-3 rounded-xl flex items-center justify-center transition-all duration-200 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed ${
                    isRecording 
                      ? 'bg-brand-red text-white animate-pulse shadow-md shadow-brand-red/15' 
                      : 'bg-dark-border text-slate-500 hover:text-slate-900 hover:bg-slate-200/50'
                  }`}
                >
                  {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                </button>
              </div>

              <div className="flex justify-end">
                <button
                  type="submit"
                  disabled={submittingLog}
                  className="bg-brand-red hover:bg-brand-red-hover text-white px-6 py-2.5 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer flex items-center space-x-2 shadow-md shadow-brand-red/10"
                >
                  {submittingLog ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Parsing Session...</span>
                    </>
                  ) : (
                    <span>LOG SESSION</span>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Today's Workout Ledger */}
          <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-sm">
            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-4">Today's Training Diary</span>
            
            <div className="space-y-3 max-h-[350px] overflow-y-auto pr-1">
              <AnimatePresence initial={false}>
                {loggedWorkouts.length > 0 ? (
                  loggedWorkouts.map((work) => (
                    <motion.div
                      key={work.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.25 }}
                      className="bg-white border border-dark-border p-4 rounded-2xl flex items-center justify-between group shadow-sm hover:border-slate-300 transition-colors duration-200"
                    >
                      <div className="flex items-center space-x-3">
                        <div className="bg-brand-red/10 text-brand-red w-8 h-8 rounded-lg flex items-center justify-center shrink-0">
                          <Calendar className="w-4 h-4" />
                        </div>
                        <div>
                          <div className="flex items-center space-x-2 mb-1.5 flex-wrap gap-y-1">
                            <button
                              onClick={() => handleOpenPRModal(work.exercise_id || work.exercise_name)}
                              className="text-sm font-bold text-slate-900 hover:text-brand-red transition-colors uppercase leading-none text-left cursor-pointer focus:outline-none"
                            >
                              {work.exercise_name}
                            </button>
                            {work.is_new_pr && (
                              <span className="text-[8px] font-black text-amber-600 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded-md uppercase tracking-wider">
                                🏆 NEW PR
                              </span>
                            )}
                          </div>
                          <span className="text-[9px] text-slate-450 font-black uppercase tracking-wider bg-slate-100 px-2 py-0.5 rounded-md">
                            {work.sets} sets x {work.reps} reps @ {work.weight}kg
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="text-right">
                          <span className="text-[8px] text-slate-400 font-bold block uppercase leading-none">Volume</span>
                          <span className="text-xs font-black text-slate-950 block mt-1">{work.volume} kg</span>
                        </div>
                        <button
                          onClick={() => handleDeleteWorkout(work.id)}
                          className="text-slate-400 hover:text-brand-red p-1 rounded transition-colors duration-200 opacity-0 group-hover:opacity-100 cursor-pointer"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <div className="text-center py-8 text-xs text-slate-400 font-semibold">
                    No exercises logged today. Say: "Bench press 3 sets of 10 reps at 60kg" to populate your diary!
                  </div>
                )}
              </AnimatePresence>
            </div>
          </div>

        </div>

      </div>

      {/* Preview Modal for confirming parsed exercises */}
      <AnimatePresence>
        {showPreview && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="bg-white rounded-3xl max-w-lg w-full max-h-[85vh] flex flex-col border border-slate-200 shadow-2xl overflow-hidden"
            >
              {/* Modal Header */}
              <div className="p-6 border-b border-slate-150 flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-black text-slate-900 m-0">Confirm Exercise Logs</h3>
                  <p className="text-xs text-slate-500 mt-1 m-0">Review and edit the parsed sets before adding to your training diary.</p>
                </div>
                <button 
                  onClick={() => setShowPreview(false)}
                  className="text-slate-400 hover:text-slate-900 text-sm font-bold bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-xl transition-all cursor-pointer"
                >
                  Cancel
                </button>
              </div>

              {/* Modal Body: Scrollable Card List */}
              <div className="p-6 overflow-y-auto space-y-4 flex-1">
                {previewLogs.map((log, idx) => (
                  <div key={idx} className="bg-slate-50/50 border border-slate-200/70 p-4 rounded-2xl space-y-3 relative">
                    <div className="flex items-center justify-between">
                      <div className="relative w-2/3">
                        <input
                          type="text"
                          value={log.exercise_name}
                          onChange={(e) => {
                            const newLogs = [...previewLogs];
                            newLogs[idx].exercise_name = e.target.value;
                            setPreviewLogs(newLogs);
                          }}
                          onFocus={() => setFocusedExerciseIdx(idx)}
                          onBlur={() => {
                            setTimeout(() => setFocusedExerciseIdx(null), 200);
                          }}
                          className="font-bold text-sm bg-transparent border-b border-transparent hover:border-slate-300 focus:border-brand-red focus:outline-none w-full py-0.5 text-slate-950 uppercase"
                          placeholder="Exercise Name"
                        />
                        
                        {focusedExerciseIdx === idx && (
                          <div className="absolute left-0 right-0 top-full mt-1 bg-white border border-slate-200 rounded-xl shadow-xl max-h-48 overflow-y-auto z-50 divide-y divide-slate-100">
                            {EXERCISE_LIST.filter(ex => 
                              ex.name.toLowerCase().includes((log.exercise_name || '').toLowerCase())
                            ).map((suggestion, sIndex) => (
                              <button
                                key={sIndex}
                                type="button"
                                onMouseDown={(e) => {
                                  e.preventDefault();
                                  const newLogs = [...previewLogs];
                                  newLogs[idx].exercise_name = suggestion.name;
                                  newLogs[idx].exercise_id = suggestion.id;
                                  newLogs[idx].muscle_group = suggestion.muscle_group;
                                  setPreviewLogs(newLogs);
                                  setFocusedExerciseIdx(null);
                                }}
                                className="w-full text-left px-3.5 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 transition-colors uppercase cursor-pointer"
                              >
                                {suggestion.name} ({suggestion.muscle_group})
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                      
                      <select
                        value={log.muscle_group}
                        onChange={(e) => {
                          const newLogs = [...previewLogs];
                          newLogs[idx].muscle_group = e.target.value;
                          setPreviewLogs(newLogs);
                        }}
                        className="text-[10px] font-black text-brand-red bg-brand-red/10 border border-brand-red/15 px-2 py-1 rounded-lg uppercase tracking-wider outline-none"
                      >
                        <option value="Chest">Chest</option>
                        <option value="Back">Back</option>
                        <option value="Legs">Legs</option>
                        <option value="Shoulders">Shoulders</option>
                        <option value="Biceps">Biceps</option>
                        <option value="General">General</option>
                      </select>
                    </div>

                    <div className="overflow-x-auto mt-2">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                            <th className="py-2 px-1 text-center w-10">Set</th>
                            <th className="py-2 px-1 text-center w-20">Reps</th>
                            <th className="py-2 px-1 text-center w-32">Weight</th>
                            <th className="py-2 px-1 text-center">Flags</th>
                            <th className="py-2 px-1 text-center w-8"></th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {log.sets && log.sets.map((setObj, sIdx) => (
                            <tr key={sIdx} className="hover:bg-slate-50/50">
                              <td className="py-2 px-1 text-center text-xs font-bold text-slate-500">
                                #{sIdx + 1}
                              </td>
                              <td className="py-2 px-1 text-center">
                                <input
                                  type="number"
                                  min="0"
                                  value={setObj.reps}
                                  onChange={(e) => {
                                    const newLogs = [...previewLogs];
                                    newLogs[idx].sets[sIdx].reps = parseInt(e.target.value) || 0;
                                    setPreviewLogs(newLogs);
                                  }}
                                  className="w-16 bg-white border border-slate-200 rounded-lg px-2 py-1 text-xs font-semibold focus:outline-none focus:border-brand-red text-slate-800 text-center"
                                />
                              </td>
                              <td className="py-2 px-1 text-center">
                                <div className="flex items-center justify-center space-x-1">
                                  <input
                                    type="number"
                                    min="0"
                                    step="0.5"
                                    value={setObj.weight_value}
                                    onChange={(e) => {
                                      const newLogs = [...previewLogs];
                                      newLogs[idx].sets[sIdx].weight_value = parseFloat(e.target.value) || 0;
                                      setPreviewLogs(newLogs);
                                    }}
                                    className="w-16 bg-white border border-slate-200 rounded-lg px-2 py-1 text-xs font-semibold focus:outline-none focus:border-brand-red text-slate-800 text-center"
                                  />
                                  <select
                                    value={setObj.weight_unit}
                                    onChange={(e) => {
                                      const newLogs = [...previewLogs];
                                      newLogs[idx].sets[sIdx].weight_unit = e.target.value;
                                      setPreviewLogs(newLogs);
                                    }}
                                    className="bg-white border border-slate-200 rounded-lg px-1 py-1 text-[10px] font-bold focus:outline-none focus:border-brand-red text-slate-700"
                                  >
                                    <option value="kg">kg</option>
                                    <option value="lbs">lbs</option>
                                  </select>
                                </div>
                              </td>
                              <td className="py-2 px-1 text-center">
                                <div className="flex items-center justify-center space-x-1.5">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      const newLogs = [...previewLogs];
                                      newLogs[idx].sets[sIdx].with_spotter = !newLogs[idx].sets[sIdx].with_spotter;
                                      setPreviewLogs(newLogs);
                                    }}
                                    className={`text-[10px] px-2 py-1 rounded-lg font-bold flex items-center space-x-0.5 border transition-all cursor-pointer ${
                                      setObj.with_spotter
                                        ? "bg-amber-500/10 text-amber-600 border-amber-500/20"
                                        : "bg-slate-100 text-slate-400 border-transparent hover:bg-slate-200"
                                    }`}
                                    title="Spotted"
                                  >
                                    <span>🤝</span>
                                    <span className="hidden sm:inline">Spotted</span>
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      const newLogs = [...previewLogs];
                                      newLogs[idx].sets[sIdx].to_failure = !newLogs[idx].sets[sIdx].to_failure;
                                      setPreviewLogs(newLogs);
                                    }}
                                    className={`text-[10px] px-2 py-1 rounded-lg font-bold flex items-center space-x-0.5 border transition-all cursor-pointer ${
                                      setObj.to_failure
                                        ? "bg-rose-500/10 text-rose-600 border-rose-500/20"
                                        : "bg-slate-100 text-slate-400 border-transparent hover:bg-slate-200"
                                    }`}
                                    title="To Failure"
                                  >
                                    <span>💀</span>
                                    <span className="hidden sm:inline">Failure</span>
                                  </button>
                                </div>
                              </td>
                              <td className="py-2 px-1 text-center">
                                <button
                                  type="button"
                                  onClick={() => {
                                    const newLogs = [...previewLogs];
                                    const currentSets = newLogs[idx].sets.filter((_, sIndex) => sIndex !== sIdx);
                                    newLogs[idx].sets = currentSets.map((s, sIndex) => ({
                                      ...s,
                                      set_number: sIndex + 1
                                    }));
                                    setPreviewLogs(newLogs);
                                  }}
                                  className="text-slate-350 hover:text-brand-red font-bold text-sm p-1 transition-colors cursor-pointer"
                                  disabled={log.sets.length <= 1}
                                  style={{ opacity: log.sets.length <= 1 ? 0.3 : 1 }}
                                  title="Remove set"
                                >
                                  ×
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Bottom controls: Volume & Add Set */}
                    <div className="flex items-center justify-between pt-3 border-t border-slate-100 mt-2">
                      <div className="text-[10px] font-bold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-lg">
                        Total Volume: <span className="text-slate-900">{log.sets ? log.sets.reduce((sum, s) => sum + (s.reps * (s.weight_value || 0)), 0) : 0}</span> {log.sets && log.sets[0]?.weight_unit || 'kg'}
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          const newLogs = [...previewLogs];
                          const currentSets = newLogs[idx].sets || [];
                          const lastSet = currentSets[currentSets.length - 1];
                          const newSet = {
                            set_number: currentSets.length + 1,
                            reps: lastSet ? lastSet.reps : 10,
                            weight_value: lastSet ? lastSet.weight_value : 0.0,
                            weight_unit: lastSet ? lastSet.weight_unit : "kg",
                            with_spotter: false,
                            to_failure: false,
                            notes: ""
                          };
                          newLogs[idx].sets = [...currentSets, newSet];
                          setPreviewLogs(newLogs);
                        }}
                        className="text-[10px] text-brand-red bg-brand-red/5 hover:bg-brand-red/10 border border-brand-red/10 px-2.5 py-1 rounded-lg font-bold transition-all flex items-center space-x-1 cursor-pointer"
                      >
                        <span>+</span>
                        <span>Add Set</span>
                      </button>
                    </div>

                    {/* Remove Card button */}
                    <button
                      type="button"
                      onClick={() => {
                        const newLogs = previewLogs.filter((_, i) => i !== idx);
                        setPreviewLogs(newLogs);
                        if (newLogs.length === 0) setShowPreview(false);
                      }}
                      className="absolute top-2 right-2 text-slate-350 hover:text-brand-red p-1 rounded transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>

              {/* Modal Footer */}
              <div className="p-6 border-t border-slate-150 flex items-center justify-between bg-slate-50/50">
                <span className="text-[10px] font-bold text-slate-400 uppercase">
                  {previewLogs.length} exercise{previewLogs.length > 1 ? 's' : ''} parsed
                </span>
                <button
                  onClick={handleConfirmLogs}
                  className="bg-brand-red hover:bg-brand-red-hover text-white px-6 py-2.5 rounded-xl font-bold text-sm tracking-wide transition-all cursor-pointer shadow-md shadow-brand-red/10"
                >
                  CONFIRM & LOG
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Toast Notification for Undo */}
      <AnimatePresence>
        {toast.visible && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-6 right-6 z-50 bg-slate-950 border border-slate-800 text-white px-5 py-4 rounded-2xl flex items-center justify-between shadow-2xl max-w-sm w-full space-x-4"
          >
            <div className="flex items-center space-x-3">
              <div className="bg-brand-red/20 text-brand-red w-8 h-8 rounded-xl flex items-center justify-center shrink-0">
                <Flame className="w-4 h-4 text-brand-red animate-pulse" />
              </div>
              <div>
                <p className="text-xs font-bold leading-none mb-1">{toast.message}</p>
                <span className="text-[10px] text-slate-400 font-medium">Click undo to reverse this action</span>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleUndoLogs}
                className="bg-white/10 hover:bg-white/20 text-white font-bold text-xs px-3 py-1.5 rounded-xl transition-all cursor-pointer"
              >
                UNDO
              </button>
              <button
                onClick={() => setToast({ visible: false, message: '', logs: [] })}
                className="text-slate-500 hover:text-slate-350 p-1 text-sm font-bold cursor-pointer"
              >
                ✕
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* PR Progress Modal */}
      <AnimatePresence>
        {prModalOpen && (
          <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="bg-slate-900 border border-slate-800 rounded-3xl max-w-lg w-full p-6 text-white shadow-2xl relative flex flex-col overflow-hidden"
              style={{ width: '480px' }}
            >
              {/* Close Button */}
              <button
                onClick={() => setPrModalOpen(false)}
                className="absolute top-4 right-4 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 w-8 h-8 rounded-full flex items-center justify-center transition-all cursor-pointer font-bold text-sm"
              >
                ✕
              </button>

              {/* Header */}
              <div className="mb-5 pr-8">
                <span className="text-[9px] font-black uppercase tracking-widest text-brand-red block mb-1">
                  Strength Progression
                </span>
                <h3 className="text-xl font-black text-white m-0 uppercase tracking-tight">
                  {prModalExercise}
                </h3>
                {prModalData && (
                  <div className="mt-2.5 flex items-center">
                    <span className="text-[10px] font-black text-amber-500 bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded-full uppercase tracking-wider flex items-center space-x-1">
                      <span>🏆 Current PR:</span>
                      <span className="text-white font-bold">{prModalData.current_pr_e1rm} kg</span>
                      <span className="text-slate-400 font-semibold normal-case">e1RM</span>
                    </span>
                  </div>
                )}
              </div>

              {/* Toggle Buttons */}
              <div className="flex bg-slate-950/60 p-1 rounded-xl mb-5 border border-slate-800/80">
                {[
                  { id: 'e1rm', label: 'Est. 1RM' },
                  { id: 'best_weight', label: 'Best Weight' },
                  { id: 'volume', label: 'Volume' }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setPrModalTab(tab.id)}
                    className={`flex-1 py-1.5 text-center text-xs font-black uppercase tracking-wider rounded-lg transition-all cursor-pointer ${
                      prModalTab === tab.id
                        ? 'bg-purple-650 text-white shadow-md'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Main Body / Graph */}
              <div className="h-[220px] flex items-center justify-center relative bg-slate-950/20 border border-slate-850/50 rounded-2xl p-4">
                {loadingPRData ? (
                  <div className="flex flex-col items-center space-y-2">
                    <Loader2 className="w-6 h-6 animate-spin text-purple-450" />
                    <p className="text-xs text-slate-500 font-semibold">Retrieving telemetry data...</p>
                  </div>
                ) : prModalData && prModalData.history && prModalData.history.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={prModalData.history}
                      margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="colorPr" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.16}/>
                          <stop offset="95%" stopColor="#a78bfa" stopOpacity={0.0}/>
                        </linearGradient>
                      </defs>
                      <XAxis 
                        dataKey="date" 
                        stroke="#475569" 
                        tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 'bold' }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis 
                        stroke="#475569"
                        tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 'bold' }}
                        axisLine={false}
                        tickLine={false}
                        domain={['dataMin - 5', 'dataMax + 5']}
                        unit=" kg"
                      />
                      <RechartsTooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const d = payload[0].payload;
                            return (
                              <div className="bg-slate-950 border border-purple-500/30 p-3 rounded-xl shadow-2xl text-[11px] text-slate-200">
                                <p className="font-bold text-slate-400 mb-1 text-[10px]">{d.date_full}</p>
                                <p className="font-black text-purple-300 flex items-center space-x-1">
                                  <span>
                                    {prModalTab === 'e1rm' && `Est. 1RM: ${d.e1rm} kg`}
                                    {prModalTab === 'best_weight' && `Best Weight: ${d.best_weight} kg`}
                                    {prModalTab === 'volume' && `Volume: ${d.volume} kg`}
                                  </span>
                                  {d.is_pr && <span className="text-amber-500 ml-1">🏆 PR</span>}
                                </p>
                                <p className="text-[9px] text-slate-500 mt-1 font-semibold">
                                  Best Set: {d.best_weight}kg × {d.best_reps} reps
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey={prModalTab}
                        stroke="#a78bfa"
                        strokeWidth={2.5}
                        fillOpacity={1}
                        fill="url(#colorPr)"
                        dot={(props) => {
                          const { cx, cy, payload } = props;
                          if (!cx || !cy) return null;
                          const isPr = payload.is_pr;
                          return (
                            <circle
                              key={props.index}
                              cx={cx}
                              cy={cy}
                              r={isPr ? 6 : 3.5}
                              fill={isPr ? "#f59e0b" : "#a78bfa"}
                              stroke={isPr ? "#f59e0b" : "transparent"}
                              strokeWidth={1.5}
                              className="cursor-pointer"
                            />
                          );
                        }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="text-center py-8 text-xs text-slate-400 font-semibold">
                    No history found for this exercise.
                  </div>
                )}
                
                {/* One data point hint */}
                {!loadingPRData && prModalData && prModalData.history && prModalData.history.length === 1 && (
                  <div className="absolute bottom-2 right-2 bg-slate-950/80 text-[9px] text-amber-500/90 font-bold px-2 py-0.5 rounded-lg border border-amber-500/10 animate-pulse">
                    Keep training to unlock trends!
                  </div>
                )}
              </div>

              {/* Footer / Caption */}
              <div className="mt-5 flex items-center justify-between text-[10px] text-slate-400 font-bold">
                <span className="flex items-center space-x-1">
                  <span>🏆</span>
                  <span>= Personal Record session</span>
                </span>
                {prModalData && prModalData.history && (
                  <span>
                    {prModalData.history.length} session{prModalData.history.length > 1 ? 's' : ''} tracked
                  </span>
                )}
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
