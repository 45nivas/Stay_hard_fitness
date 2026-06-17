import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Camera, 
  Upload, 
  Sparkles, 
  Loader2, 
  AlertTriangle, 
  Activity, 
  TrendingUp, 
  Compass 
} from 'lucide-react';
import { 
  Radar, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  ResponsiveContainer 
} from 'recharts';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export default function BodyVision() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    setError('');
    const file = e.target.files[0];
    if (!file) return;

    // Validate type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setError('Please upload a JPG, PNG, or WEBP photo only.');
      return;
    }

    // Validate size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size exceeds the 10MB limit.');
      return;
    }

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setError('');
    setLoading(true);

    const formData = new FormData();
    formData.append('photo', selectedFile);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/analyse-body/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (res.data.success) {
        setResults(res.data.analysis);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || 'Analysis failed. Make sure your chest, shoulders, and hips are clearly visible.');
    } finally {
      setLoading(false);
    }
  };

  // Convert muscle scores JSON to Recharts format
  const getRadarData = (muscleScores) => {
    if (!muscleScores) return [];
    return Object.entries(muscleScores).map(([muscle, score]) => ({
      subject: muscle.replace('_', ' ').toUpperCase(),
      score: score,
      fullMark: 10
    }));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-3xl font-extrabold tracking-tight text-white m-0">Body Vision Analyser</h2>
        <p className="text-gray-400 text-sm mt-1">Upload a front-facing physique photo to extract muscle proportions, skeletal tapers, and balancing parameters.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Upload Form Block */}
        <div className="lg:col-span-1 bg-dark-card border border-dark-border p-6 rounded-3xl h-fit shadow-lg space-y-5">
          <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
            <Camera className="w-5 h-5 text-brand-red" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Photo Uploader</h3>
          </div>

          {error && (
            <div className="bg-brand-red/10 border border-brand-red/50 text-brand-red text-xs p-3 rounded-lg text-center font-bold">
              {error}
            </div>
          )}

          <form onSubmit={handleUploadSubmit} className="space-y-5">
            {/* File Drag Box */}
            <div className="relative border-2 border-dashed border-dark-border hover:border-brand-red rounded-2xl overflow-hidden aspect-[3/4] flex items-center justify-center bg-dark-bg cursor-pointer group transition-all duration-200">
              <input 
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 opacity-0 cursor-pointer z-20"
              />
              
              {previewUrl ? (
                <img 
                  src={previewUrl} 
                  alt="Preview"
                  className="w-full h-full object-cover z-10"
                />
              ) : (
                <div className="text-center p-4 space-y-2 select-none z-10">
                  <Upload className="w-10 h-10 text-gray-500 group-hover:text-brand-red mx-auto transition-colors duration-200" />
                  <p className="text-xs font-bold text-white">Upload shirtless photo</p>
                  <p className="text-[10px] text-gray-500 uppercase tracking-widest">JPG, PNG or WEBP (Max 10MB)</p>
                </div>
              )}
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading || !selectedFile}
              className="w-full bg-brand-red hover:bg-brand-red-hover text-white py-3.5 rounded-xl font-bold text-sm tracking-wide transition-all duration-200 cursor-pointer disabled:opacity-50 flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Scanning Physique...</span>
                </>
              ) : (
                <span>ANALYSE SYMMETRY</span>
              )}
            </button>
          </form>
        </div>

        {/* Results Block */}
        <div className="lg:col-span-2">
          {results ? (
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {/* Muscle Balance Radar Chart */}
              <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg flex flex-col md:flex-row gap-6">
                
                {/* Chart Frame */}
                <div className="w-full md:w-1/2 aspect-square flex items-center justify-center bg-dark-bg/50 border border-dark-border/50 rounded-2xl p-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="75%" data={getRadarData(results.muscle_scores)}>
                      <PolarGrid stroke="#2e2e2e" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 9, fontWeight: 'bold' }} />
                      <PolarRadiusAxis angle={30} domain={[0, 10]} tick={{ fill: '#4b5563' }} />
                      <Radar
                        name="Development"
                        dataKey="score"
                        stroke="#e50914"
                        fill="#e50914"
                        fillOpacity={0.4}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>

                {/* Muscle metrics lists */}
                <div className="flex-1 flex flex-col justify-between space-y-4">
                  <div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest block">Physique Classification</span>
                    <h3 className="text-2xl font-black text-white mt-1 uppercase select-none">
                      {results.taper_assessment || 'Athletic'} • {results.body_type || 'Mesomorph'}
                    </h3>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-widest">Dominant Groups</span>
                      <div className="flex flex-wrap gap-1.5 mt-1.5">
                        {results.dominant_groups.map(g => (
                          <span key={g} className="bg-dark-bg border border-dark-border text-white text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full">
                            {g}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div>
                      <span className="text-[9px] font-bold text-brand-red uppercase tracking-widest">Weak / Lagging Groups</span>
                      <div className="flex flex-wrap gap-1.5 mt-1.5">
                        {results.weak_groups.map(g => (
                          <span key={g} className="bg-brand-red/10 border border-brand-red/30 text-brand-red text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-full">
                            {g}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Landmark Ratios Grid */}
                  {results.landmark_data && (
                    <div className="grid grid-cols-3 gap-2 text-center bg-dark-bg border border-dark-border/60 p-3.5 rounded-xl">
                      <div>
                        <span className="text-[8px] text-gray-500 font-bold uppercase block">Taper Ratio</span>
                        <p className="text-sm font-black text-white mt-0.5">{results.landmark_data.taper_ratio?.toFixed(2) || '1.0'}</p>
                      </div>
                      <div className="border-x border-dark-border">
                        <span className="text-[8px] text-gray-500 font-bold uppercase block">Arm Symmetry</span>
                        <p className="text-sm font-black text-white mt-0.5">{results.landmark_data.arm_symmetry_score?.toFixed(1) || '100'}%</p>
                      </div>
                      <div>
                        <span className="text-[8px] text-gray-500 font-bold uppercase block">Leg Symmetry</span>
                        <p className="text-sm font-black text-white mt-0.5">{results.landmark_data.leg_symmetry_score?.toFixed(1) || '100'}%</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Recommendation Split */}
              <div className="bg-dark-card border border-dark-border p-6 rounded-3xl shadow-lg space-y-4">
                <div className="flex items-center space-x-2 border-b border-dark-border pb-3">
                  <TrendingUp className="w-5 h-5 text-brand-red" />
                  <h3 className="text-sm font-bold uppercase tracking-wider text-white m-0">Recommended Programming</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="md:col-span-1 bg-dark-bg border border-dark-border p-4 rounded-xl text-center flex flex-col justify-center">
                    <span className="text-[8px] text-gray-500 font-bold uppercase block">Suggested Split</span>
                    <p className="text-sm font-black text-brand-red uppercase mt-1 leading-tight">{results.suggested_split}</p>
                  </div>

                  <div className="md:col-span-2 space-y-1">
                    <span className="text-[8px] text-gray-500 font-bold uppercase block">Priority Recommendations</span>
                    <p className="text-xs text-gray-300 leading-relaxed font-semibold italic mt-1">
                      "{results.priority_recommendation}"
                    </p>
                  </div>
                </div>
              </div>

            </motion.div>
          ) : (
            <div className="bg-dark-card border border-dark-border border-dashed p-12 rounded-3xl text-center text-gray-500 text-sm font-semibold h-full flex flex-col justify-center items-center space-y-3 min-h-[40vh]">
              <Compass className="w-8 h-8 text-gray-600 animate-spin" />
              <p>Upload a shirtless posture snapshot to calculate skeletal tapering and muscular density balance.</p>
            </div>
          )}
        </div>

      </div>
    </motion.div>
  );
}
