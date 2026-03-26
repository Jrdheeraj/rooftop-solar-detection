import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const messages = [
  "Analyzing rooftop satellite data...",
  "Applying YOLOv8 detection model...",
  "Differentiating panels from structures...",
  "Calculating surface coverage...",
  "Estimating energy potential...",
  "Finalizing your report..."
];

const Loader = ({ message }) => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % messages.length);
    }, 2500);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="fixed inset-0 z-[100] bg-white/95 backdrop-blur-xl flex flex-col items-center justify-center p-6 text-center">
      <div className="relative w-24 h-24 mb-12">
        {/* Core pulse */}
        <motion.div 
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="absolute inset-0 bg-green-500/20 rounded-full"
        />
        {/* Spinning border */}
        <motion.div 
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
          className="absolute inset-0 border-t-4 border-r-4 border-green-500 rounded-full"
        />
        {/* Inner static icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-4 h-4 bg-green-600 rounded-full shadow-[0_0_15px_rgba(22,163,74,0.6)]" />
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.5 }}
          className="space-y-4"
        >
          <h3 className="text-2xl font-black text-gray-900 tracking-tight">
            Processing Data
          </h3>
          <p className="text-[#16a34a] font-mono text-sm font-bold tracking-widest uppercase bg-green-50 px-4 py-1.5 rounded-full inline-block">
            {message || messages[index]}
          </p>
        </motion.div>
      </AnimatePresence>

      <div className="mt-16 w-64 h-1 bg-gray-100 rounded-full overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ duration: 15, ease: "linear" }}
          className="h-full bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]"
        />
      </div>
      
      <p className="mt-4 text-xs font-bold text-gray-400 uppercase tracking-[0.2em]">
        Model Version v2.4.1 · High Intensity Scan
      </p>
    </div>
  );
};

export default Loader;
