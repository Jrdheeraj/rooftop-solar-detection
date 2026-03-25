import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiMapPin, FiCpu, FiGrid, FiTarget, FiDownload, 
  FiCode, FiArrowLeft, FiActivity, FiSearch, 
  FiCornerDownRight, FiCheckCircle, FiAlertCircle
} from 'react-icons/fi';

const ResultsSection = ({ data, onReset }) => {
  const [showJson, setShowJson] = useState(false);

  // Fallback dummy data if none provided
  const result = data || {
    overlay_path: "https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
    has_solar: true,
    confidence: 0.947,
    panels_in_buffer: 12,
    total_area: 42.8,
    lat: 17.4482,
    lng: 78.3915,
    panels: [
      { id: 1, confidence: 0.98, area: 2.1 },
      { id: 2, confidence: 0.97, area: 2.2 },
      { id: 3, confidence: 0.94, area: 2.0 },
      { id: 4, confidence: 0.99, area: 2.3 },
    ]
  };

  return (
    <section className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        
        {/* Header Actions */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-12">
          <div>
            <motion.button 
              onClick={onReset}
              whileHover={{ x: -4 }}
              className="flex items-center gap-2 text-sm font-bold text-gray-500 hover:text-black transition-colors mb-4 uppercase tracking-wider"
            >
              <FiArrowLeft /> Back to Input
            </motion.button>
            <h2 className="text-4xl font-black text-gray-900 tracking-tight">
              Analysis <span className="text-[#16a34a]">Complete</span>
            </h2>
          </div>
          <div className="flex gap-3">
            <button className="flex items-center gap-2 px-6 py-3 border border-gray-200 rounded-2xl text-sm font-bold text-gray-700 hover:bg-gray-50 transition-all">
              <FiDownload /> Download Data
            </button>
            <button className="flex items-center gap-2 px-6 py-3 bg-[#16a34a] text-white rounded-2xl text-sm font-bold shadow-lg shadow-green-600/20 hover:bg-[#15803d] transition-all">
              <FiCheckCircle /> Report Ready
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT: Overlay Viewing Area */}
          <div className="lg:col-span-7">
            <div className="sticky top-12">
              <div className="relative bg-gray-900 rounded-[2.5rem] overflow-hidden shadow-2xl group border-[12px] border-gray-50">
                <img 
                  src={result.overlay_path} 
                  alt="Detection Overlay" 
                  className="w-full h-auto object-cover opacity-90 group-hover:scale-105 transition-transform duration-700"
                />
                <div className="absolute top-6 left-6 flex flex-wrap gap-2">
                  <span className="bg-black/40 backdrop-blur-md text-white px-4 py-2 rounded-full text-xs font-bold flex items-center gap-2 border border-white/20">
                    <FiSearch className="text-green-400" /> AI Detection Overlay
                  </span>
                  <span className="bg-green-500 text-white px-4 py-2 rounded-full text-xs font-bold shadow-lg">
                    {result.panels_in_buffer} Panels Found
                  </span>
                </div>
                <div className="absolute bottom-6 right-6">
                   <div className="bg-white/90 backdrop-blur-md p-4 rounded-3xl shadow-xl flex items-center gap-4 border border-white">
                      <div className="w-10 h-10 bg-green-100 rounded-2xl flex items-center justify-center text-green-600">
                        <FiActivity className="text-xl" />
                      </div>
                      <div>
                        <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Confidence</p>
                        <p className="text-lg font-bold text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
                      </div>
                   </div>
                </div>
              </div>
              <p className="mt-6 text-sm text-gray-400 font-medium text-center">
                Visual representation of bounding boxes and segmented panel areas.
              </p>
            </div>
          </div>

          {/* RIGHT: Dashboard Panels */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Main Stats Card */}
            <div className="bg-[#f8faf8] rounded-[2rem] p-8 border border-green-100/50">
              <div className="flex items-center gap-2 text-xs font-black text-green-700 uppercase tracking-[0.2em] mb-6">
                <FiCpu /> Core Intelligence
              </div>
              <div className="grid grid-cols-2 gap-8">
                <div>
                  <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Status</p>
                  <p className={`text-xl font-bold ${result.has_solar ? 'text-green-600' : 'text-red-500'}`}>
                    {result.has_solar ? 'Solar Detected' : 'No Solar'}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Area Estimate</p>
                  <p className="text-xl font-bold text-gray-900">{result.total_area} sq.m</p>
                </div>
                <div>
                   <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Latitude</p>
                   <p className="text-base font-bold text-gray-900">{result.lat || 'N/A'}</p>
                </div>
                <div>
                   <p className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Longitude</p>
                   <p className="text-base font-bold text-gray-900">{result.lng || 'N/A'}</p>
                </div>
              </div>
            </div>

            {/* Individual Panels Card */}
            <div className="bg-white border border-gray-100 rounded-[2rem] p-8 shadow-sm">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-2 text-xs font-black text-gray-400 uppercase tracking-[0.2em]">
                  <FiGrid /> Panel Distribution
                </div>
                <span className="text-[10px] font-bold bg-gray-100 px-2 py-1 rounded text-gray-500 uppercase">
                  Sort: Confidence
                </span>
              </div>
              
              <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                {result.panels?.map((panel, idx) => (
                  <div key={idx} className="group flex items-center justify-between p-4 rounded-2xl bg-gray-50/50 border border-transparent hover:border-green-200 hover:bg-green-50/20 transition-all">
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 rounded-lg bg-white shadow-sm flex items-center justify-center text-[10px] font-black text-gray-400">
                        {idx + 1}
                      </div>
                      <div>
                        <p className="text-xs font-bold text-gray-800">Panel ID #{panel.id}</p>
                        <p className="text-[10px] font-medium text-gray-400 uppercase">{panel.area}m² Segment</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="w-24 h-1.5 bg-gray-200 rounded-full overflow-hidden mb-1">
                        <div 
                          className="h-full bg-green-500" 
                          style={{ width: `${panel.confidence * 100}%` }} 
                        />
                      </div>
                      <p className="text-[10px] font-black text-green-600">{(panel.confidence * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* JSON Toggle */}
            <div className="pt-4">
              <button 
                onClick={() => setShowJson(!showJson)}
                className="w-full flex items-center justify-between p-5 bg-gray-50 rounded-2xl text-xs font-bold text-gray-500 hover:text-black transition-all border border-gray-100"
              >
                <span className="flex items-center gap-2"><FiCode /> Development Metadata (JSON)</span>
                <FiCornerDownRight className={`transition-transform ${showJson ? 'rotate-90' : ''}`} />
              </button>
              <AnimatePresence>
                {showJson && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden mt-2"
                  >
                    <pre className="p-6 bg-gray-900 text-green-400 rounded-2xl text-[10px] font-mono whitespace-pre-wrap leading-relaxed shadow-inner">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

          </div>
        </div>
      </div>
    </section>
  );
};

export default ResultsSection;
