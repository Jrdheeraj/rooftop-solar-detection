import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FiUploadCloud, FiMapPin, FiMaximize, FiTarget, FiCheckCircle, FiArrowUpRight } from 'react-icons/fi';

const InputSection = ({ onAnalyze }) => {
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [coords, setCoords] = useState({ lat: '', lng: '' });
  const [uploadCoords, setUploadCoords] = useState({ lat: '', lng: '' });
  const [uploadImageType, setUploadImageType] = useState('PHOTO');
  const [confidence, setConfidence] = useState(0.5);
  const [buffer, setBuffer] = useState('1200');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleSubmit = () => {
    if (activeTab === 'upload' && !file) return alert('Please upload an image');
    if (activeTab === 'coords' && (!coords.lat || !coords.lng)) {
      return alert('Please enter valid coordinates');
    }

    if (activeTab === 'upload' && uploadImageType === 'SATELLITE') {
      const lat = parseFloat(uploadCoords.lat);
      const lng = parseFloat(uploadCoords.lng);
      if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        return alert('Satellite image uploads need valid latitude and longitude for accurate scaling');
      }
    }
    
    // Validate coordinates
    if (activeTab === 'coords') {
      const lat = parseFloat(coords.lat);
      const lng = parseFloat(coords.lng);
      if (isNaN(lat) || isNaN(lng) || lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        return alert('Please enter valid latitude (-90 to 90) and longitude (-180 to 180)');
      }
    }
    
    onAnalyze({
      type: activeTab,
      data: activeTab === 'upload' ? file : coords,
      imageType: uploadImageType,
      latitude: uploadCoords.lat,
      longitude: uploadCoords.lng,
      confidence,
      buffer
    });
  };

  return (
    <section id="analyze" className="py-24 bg-[#f8faf8] relative overflow-hidden">
      {/* Decorative background blobs */}
      <div className="absolute top-0 left-0 w-96 h-96 bg-green-100 rounded-full mix-blend-multiply filter blur-3xl opacity-30 -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-100 rounded-full mix-blend-multiply filter blur-3xl opacity-30 translate-x-1/2 translate-y-1/2" />

      <div className="max-w-4xl mx-auto px-6 relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl font-extrabold text-[#111] mb-4 tracking-tight">
            Start Your <span className="text-[#16a34a]">Analysis</span>
          </h2>
          <p className="text-gray-500 text-lg max-w-xl mx-auto">
            Upload an aerial shot or drop coordinates to identify solar potential in seconds.
          </p>
        </motion.div>

        {/* Main Card */}
        <div className="bg-white rounded-[2.5rem] shadow-2xl shadow-green-900/5 border border-gray-100 overflow-hidden">
          {/* Tabs Navigation */}
          <div className="flex bg-gray-50/50 p-2 gap-2 border-bottom border-gray-100">
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-3xl text-sm font-semibold transition-all ${
                activeTab === 'upload' 
                ? 'bg-white text-[#16a34a] shadow-sm ring-1 ring-black/5' 
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100/50'
              }`}
            >
              <FiUploadCloud className="text-lg text-current" />
              Upload Image
            </button>
            <button
              onClick={() => setActiveTab('coords')}
              className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-3xl text-sm font-semibold transition-all ${
                activeTab === 'coords' 
                ? 'bg-white text-[#16a34a] shadow-sm ring-1 ring-black/5' 
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100/50'
              }`}
            >
              <FiMapPin className="text-lg text-current" />
              Enter Coordinates
            </button>
          </div>

          <div className="p-8 md:p-12">
            <AnimatePresence mode="wait">
              {activeTab === 'upload' ? (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 10 }}
                  className="space-y-8"
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Image Type</label>
                      <select
                        value={uploadImageType}
                        onChange={(e) => setUploadImageType(e.target.value)}
                        className="w-full px-6 py-4 bg-gray-50 border border-gray-100 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-500/20 font-semibold text-gray-700"
                      >
                        <option value="PHOTO">Normal Image / Photo</option>
                        <option value="SATELLITE">Satellite Image</option>
                      </select>
                    </div>
                    <div className="p-5 bg-green-50/60 rounded-2xl border border-green-100 flex items-start gap-3">
                      <div className="p-2 bg-green-100 rounded-lg text-green-600 mt-0.5"><FiTarget /></div>
                      <p className="text-sm text-green-800 leading-relaxed font-medium">
                        Use <span className="font-bold">Photo</span> for normal rooftop images. Use <span className="font-bold">Satellite</span> only for top-down map or satellite images.
                      </p>
                    </div>
                  </div>

                  <label className="relative group cursor-pointer block border-2 border-dashed border-gray-200 rounded-[2rem] p-12 text-center hover:border-green-400 hover:bg-green-50/30 transition-all">
                    <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                    
                    {preview ? (
                      <div className="relative inline-block">
                        <img src={preview} alt="Preview" className="max-h-64 rounded-2xl shadow-lg border-4 border-white mx-auto" />
                        <div className="mt-4 text-sm font-medium text-green-600 flex items-center justify-center gap-1">
                          <FiCheckCircle /> Change Image
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="w-16 h-16 bg-green-50 rounded-2xl flex items-center justify-center mx-auto text-green-600 group-hover:scale-110 transition-transform">
                          <FiUploadCloud className="text-3xl" />
                        </div>
                        <div>
                          <p className="text-lg font-bold text-gray-800">Drop your rooftop image</p>
                          <p className="text-sm text-gray-500 mt-1">PNG, JPG or JPEG up to 10MB</p>
                        </div>
                        <span className="inline-block px-6 py-2 bg-black text-white text-xs font-bold rounded-full uppercase tracking-widest">
                          Browse Files
                        </span>
                      </div>
                    )}
                  </label>

                  {uploadImageType === 'SATELLITE' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Latitude</label>
                        <input
                          type="text"
                          placeholder="e.g. 17.4483"
                          value={uploadCoords.lat}
                          onChange={(e) => setUploadCoords({ ...uploadCoords, lat: e.target.value })}
                          className="w-full px-6 py-4 bg-gray-50 border border-gray-100 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all font-medium"
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Longitude</label>
                        <input
                          type="text"
                          placeholder="e.g. 78.3915"
                          value={uploadCoords.lng}
                          onChange={(e) => setUploadCoords({ ...uploadCoords, lng: e.target.value })}
                          className="w-full px-6 py-4 bg-gray-50 border border-gray-100 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all font-medium"
                        />
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="coords"
                  initial={{ opacity: 0, x: 10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="space-y-6"
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Latitude</label>
                      <input 
                        type="text" 
                        placeholder="e.g. 17.4483"
                        value={coords.lat}
                        onChange={(e) => setCoords({...coords, lat: e.target.value})}
                        className="w-full px-6 py-4 bg-gray-50 border border-gray-100 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all font-medium"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-gray-400 uppercase tracking-wider ml-1">Longitude</label>
                      <input 
                        type="text" 
                        placeholder="e.g. 78.3915"
                        value={coords.lng}
                        onChange={(e) => setCoords({...coords, lng: e.target.value})}
                        className="w-full px-6 py-4 bg-gray-50 border border-gray-100 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all font-medium"
                      />
                    </div>
                  </div>
                  <div className="p-6 bg-blue-50/50 rounded-2xl border border-blue-100 flex items-start gap-4">
                    <div className="p-2 bg-blue-100 rounded-lg text-blue-600 mt-1"><FiMapPin /></div>
                    <p className="text-sm text-blue-800 leading-relaxed font-medium transition-transform">
                      Coordinates will be used to fetch the latest high-resolution satellite tiles for detection.
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Common Controls */}
            <div className="mt-12 pt-10 border-t border-gray-50 grid grid-cols-1 md:grid-cols-2 gap-10">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-bold text-gray-800 flex items-center gap-2">
                    <FiTarget className="text-green-600" /> Confidence Threshold
                  </label>
                  <span className="text-xs font-black bg-green-100 text-green-700 px-2 py-1 rounded-md">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <input 
                  type="range" 
                  min="0.1" 
                  max="1.0" 
                  step="0.05" 
                  value={confidence}
                  onChange={(e) => setConfidence(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-[#16a34a]" 
                />
              </div>

              <div className="space-y-4">
                <label className="text-sm font-bold text-gray-800 flex items-center gap-2">
                  <FiMaximize className="text-green-600" /> Buffer Size
                </label>
                <select 
                  value={buffer}
                  onChange={(e) => setBuffer(e.target.value)}
                  className="w-full px-6 py-4 bg-gray-50 border border-gray-100 rounded-2xl focus:outline-none focus:ring-2 focus:ring-green-500/20 font-semibold text-gray-700"
                >
                  <option value="1200">1200 sq.ft (Standard)</option>
                  <option value="2400">2400 sq.ft (Large)</option>
                  <option value="4800">4800 sq.ft (Industrial)</option>
                </select>
              </div>
            </div>

            {/* CTA Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSubmit}
              className="w-full mt-12 py-5 bg-[#16a34a] hover:bg-[#15803d] text-white rounded-3xl font-bold text-lg shadow-xl shadow-green-600/20 transition-all flex items-center justify-center gap-3 group"
            >
              Analyze Rooftop
              <FiArrowUpRight className="text-xl group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
            </motion.button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default InputSection;
