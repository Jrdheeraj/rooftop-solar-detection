import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiMapPin, FiCpu, FiGrid, FiTarget, FiDownload, 
  FiCode, FiArrowLeft, FiActivity, FiSearch, 
  FiCornerDownRight, FiCheckCircle, FiAlertCircle, FiZap,
  FiDollarSign, FiTrendingUp, FiWind, FiSun, FiLayers, FiFileText
} from 'react-icons/fi';

const ResultsSection = ({ data, onReset }) => {
  const [showJson, setShowJson] = useState(false);

  // Transform the new API response format to match our UI needs
  const result = {
    overlay_path: data?.overlay_path || "https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
    has_solar: data?.has_solar || false,
    confidence: data?.confidence || 0,
    panels_in_buffer: data?.panels_in_buffer?.length || 0,
    total_area: data?.pv_area_sqm_est || 0,
    estimated_capacity_kw: data?.estimated_capacity_kw || 0,
    estimated_annual_production_kwh: data?.estimated_annual_production_kwh || 0,
    lat: data?.latitude || data?.lat || null,
    lng: data?.longitude || data?.lng || null,
    panels: data?.panels_in_buffer?.map((panel, idx) => ({
      id: panel.panel_id || idx + 1,
      confidence: panel.conf || 0,
      area: panel.inside_area_sqm || panel.full_area_sqm || 0,
      full_area_sqm: panel.full_area_sqm || 0,
      inside_area_sqm: panel.inside_area_sqm || 0,
      overlap_ratio: panel.overlap_ratio || 0,
      bbox_center: panel.bbox_center || []
    })) || [],
    qc_status: data?.qc_status || 'NOT_VERIFIABLE',
    buffer_radius_sqft: data?.buffer_radius_sqft || 1200,
    best_panel_id: data?.best_panel_id || -1,
    bbox_or_mask: data?.bbox_or_mask || '',
    image_metadata: data?.image_metadata || {},
    // New Metrics
    financial: data?.financial_insights || {
      est_installation_cost: 0,
      payback_years: 0,
      lifetime_savings_25yr: 0
    },
    environmental: data?.environmental_impact || {
      co2_saved_tons_yr: 0,
      trees_planted_equiv: 0,
      ev_miles_equiv: 0
    },
    technical: data?.technical_specs || {
      irradiance_kwh_m2_day: 0,
      recommended_inverter_kw: 0,
      potential_storage_kwh: 0
    }
  };

  return (
    <section className="py-24 bg-[#fafbfc]">
      <div className="max-w-7xl mx-auto px-6">
        
        {/* TOP COMPACT HEADER */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-12">
          <div>
            <motion.button 
              onClick={onReset}
              className="flex items-center gap-2 text-[10px] font-black text-gray-400 hover:text-black transition-colors mb-2 uppercase tracking-[0.2em]"
            >
              <FiArrowLeft /> Back to Scan
            </motion.button>
            <h2 className="text-3xl font-black text-gray-900 tracking-tight">
              SolarScan <span className="text-[#16a34a]">Intelligence Report</span>
            </h2>
          </div>
          <div className="flex gap-3">
            <button className="flex items-center gap-2 px-5 py-3 border border-gray-200 rounded-2xl text-xs font-bold text-gray-700 hover:bg-gray-50 transition-all">
              <FiFileText /> Export PDF
            </button>
            <button className="flex items-center gap-2 px-5 py-3 bg-[#16a34a] text-white rounded-2xl text-xs font-bold shadow-lg shadow-green-600/20 hover:bg-[#15803d] transition-all">
              <FiZap /> Consult Specialist
            </button>
          </div>
        </div>

        {/* MAIN DASHBOARD CONTENT */}
        <div className="space-y-8">
          
          {/* 1. HERO IMAGE (CENTRAL) */}
          <div className="w-full">
            <div className="relative bg-black rounded-[2.5rem] overflow-hidden shadow-2xl group border-4 border-slate-800 ring-1 ring-white/10 mx-auto max-w-screen-2xl">
              <img 
                src={result.overlay_path} 
                alt="Detection Overlay" 
                className="w-full h-auto object-cover opacity-95 shadow-[0_0_50px_rgba(0,0,0,0.5)]"
              />
              {/* Overlay Badges */}
              <div className="absolute top-6 left-6 flex flex-wrap gap-2">
                <span className="bg-black/60 backdrop-blur-xl text-white px-4 py-1.5 rounded-full text-[9px] font-black flex items-center gap-2 border border-white/20 uppercase tracking-widest">
                  <FiSearch className="text-green-400" /> AI Detection
                </span>
                <span className={`${result.has_solar ? 'bg-green-600' : 'bg-red-600'} text-white px-4 py-1.5 rounded-full text-[9px] font-black shadow-xl uppercase tracking-widest`}>
                  {result.panels_in_buffer} Panels Detected
                </span>
                <span className="bg-blue-600 text-white px-4 py-1.5 rounded-full text-[9px] font-black shadow-xl uppercase tracking-widest">
                  {result.qc_status}
                </span>
              </div>
              
              {/* Floating Confidence Indicator */}
              <div className="absolute bottom-4 right-4 scale-75 origin-bottom-right">
                <div className="bg-white/95 backdrop-blur-md px-4 py-3 rounded-3xl shadow-2xl flex items-center gap-4 border border-white/50">
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                    result.has_solar ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                  }`}>
                    <FiActivity className="text-xl" />
                  </div>
                  <div>
                    <p className="text-[9px] font-black text-gray-400 uppercase tracking-widest">Model Confidence</p>
                    <p className="text-xl font-black text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 2. KPI QUICK STATS GRID */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
             {[
               { icon: <FiGrid />, label: 'Total PV Area', value: `${result.total_area.toFixed(1)} sq.m`, color: 'text-gray-900' },
               { icon: <FiZap />, label: 'System Capacity', value: `${result.estimated_capacity_kw.toFixed(2)} kW`, color: 'text-blue-600' },
               { icon: <FiTrendingUp />, label: 'Annual Yield', value: `${result.estimated_annual_production_kwh.toLocaleString()} kWh`, color: 'text-green-600' },
               { icon: <FiDollarSign />, label: 'Payback Est.', value: `${result.financial.payback_years} Years`, color: 'text-orange-600' }
             ].map((stat, i) => (
               <div key={i} className="bg-white p-6 rounded-[2rem] border border-gray-100 shadow-sm flex items-center gap-5">
                 <div className="w-12 h-12 bg-gray-50 rounded-2xl flex items-center justify-center text-xl text-gray-400">
                   {stat.icon}
                 </div>
                 <div>
                   <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">{stat.label}</p>
                   <p className={`text-xl font-black ${stat.color}`}>{stat.value}</p>
                 </div>
               </div>
             ))}
          </div>

          {/* 3. MULTI-INSIGHTS GRID */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* FINANCIAL INSIGHTS */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-sm relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-32 h-32 bg-green-50 rounded-full -mr-16 -mt-16" />
              <div className="relative">
                <div className="flex items-center gap-3 text-xs font-black text-green-700 uppercase tracking-[0.2em] mb-8">
                  <FiDollarSign className="text-xl" /> Financial Insights
                </div>
                <div className="space-y-6">
                  <div>
                    <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-2">Estimated Investment</p>
                    <p className="text-3xl font-black text-gray-900">${result.financial.est_installation_cost.toLocaleString()}</p>
                    <p className="text-[10px] font-bold text-gray-400 mt-1">Ref. Industry Average: $1.2/W</p>
                  </div>
                  <div className="pt-6 border-t border-gray-50">
                    <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-2">25-Year Lifetime Savings</p>
                    <p className="text-3xl font-black text-[#16a34a]">${result.financial.lifetime_savings_25yr.toLocaleString()}</p>
                    <p className="text-[10px] font-bold text-green-600 mt-1">ROI: 350%+</p>
                  </div>
                </div>
              </div>
            </div>

            {/* ENVIRONMENTAL IMPACT */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-sm relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 rounded-full -mr-16 -mt-16" />
              <div className="relative">
                <div className="flex items-center gap-3 text-xs font-black text-blue-700 uppercase tracking-[0.2em] mb-8">
                  <FiWind className="text-xl" /> Eco Footprint
                </div>
                <div className="space-y-8">
                  <div className="flex items-center gap-5">
                    <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center text-lg">
                      <FiZap />
                    </div>
                    <div>
                      <p className="text-lg font-black text-gray-900">{result.environmental.co2_saved_tons_yr} Tons</p>
                      <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">CO2 Offset / Year</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-5">
                    <div className="w-10 h-10 bg-green-50 text-green-600 rounded-xl flex items-center justify-center text-lg">
                      <FiSun />
                    </div>
                    <div>
                      <p className="text-lg font-black text-gray-900">{result.environmental.trees_planted_equiv.toLocaleString()}</p>
                      <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Trees Equivalent</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-5">
                    <div className="w-10 h-10 bg-orange-50 text-orange-600 rounded-xl flex items-center justify-center text-lg">
                      <FiActivity />
                    </div>
                    <div>
                      <p className="text-lg font-black text-gray-900">{result.environmental.ev_miles_equiv.toLocaleString()}</p>
                      <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">EV Miles Avoided</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* TECHNICAL SPECS */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-sm">
              <div className="flex items-center gap-3 text-xs font-black text-gray-400 uppercase tracking-[0.2em] mb-8">
                <FiCpu className="text-xl" /> System Technicals
              </div>
              <div className="space-y-6">
                 <div>
                   <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Recommended Inverter</p>
                   <p className="text-xl font-black text-gray-900">{result.technical.recommended_inverter_kw} kW Hybrid</p>
                 </div>
                 <div>
                   <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Storage Potential</p>
                   <p className="text-xl font-black text-gray-900">{result.technical.potential_storage_kwh} kWh Lithium</p>
                 </div>
                 <div>
                   <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Grid Compliance</p>
                   <div className="flex items-center gap-2 mt-1">
                     <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                     <p className="text-sm font-bold text-gray-900">Net Metering Compatible</p>
                   </div>
                 </div>
                 <div className="pt-4 mt-6 border-t border-gray-50 flex items-center justify-between">
                    <div>
                      <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Location Irradiance</p>
                      <p className="text-sm font-bold text-gray-900">{result.technical.irradiance_kwh_m2_day} kWh/m²/day</p>
                    </div>
                    <FiMapPin className="text-gray-300 text-2xl" />
                 </div>
              </div>
            </div>
          </div>

          {/* 4. PANEL DISTRIBUTION & JSON (SIDE BY SIDE) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
             {/* Existing Panel Distribution Feature */}
             <div className="bg-white border border-gray-100 rounded-[2.5rem] p-8 shadow-sm">
               <div className="flex justify-between items-center mb-8">
                 <div className="flex items-center gap-3 text-xs font-black text-gray-400 uppercase tracking-[0.2em]">
                   <FiLayers className="text-xl" /> Precise Panel Breakdown
                 </div>
                 <span className="text-[10px] font-bold bg-gray-50 px-3 py-1.5 rounded-full text-gray-400 uppercase border border-gray-100">
                   {result.panels.length} Total detected
                 </span>
               </div>
               
               <div className="space-y-4 max-h-[400px] overflow-y-auto pr-3 custom-scrollbar">
                 {result.panels.length > 0 ? (
                   result.panels.map((panel, idx) => (
                     <div key={idx} className={`group flex items-center justify-between p-5 rounded-3xl bg-gray-50/50 border border-gray-100 ${
                       panel.id === result.best_panel_id ? 'ring-2 ring-green-400 bg-green-50/30 border-transparent shadow-lg shadow-green-100' : ''
                     }`}>
                       <div className="flex items-center gap-5">
                         <div className={`w-10 h-10 rounded-2xl shadow-sm flex items-center justify-center text-xs font-black ${
                           panel.id === result.best_panel_id 
                             ? 'bg-green-500 text-white' 
                             : 'bg-white text-gray-400 border border-gray-100'
                         }`}>
                           #{String(panel.id + 1).padStart(2, '0')}
                         </div>
                         <div>
                           <p className="text-sm font-black text-gray-900 flex items-center gap-2">
                             Panel Identity
                             {panel.id === result.best_panel_id && (
                               <span className="text-[8px] bg-green-500 text-white px-2 py-0.5 rounded-full uppercase tracking-tighter">Prime</span>
                             )}
                           </p>
                           <p className="text-[10px] font-bold text-gray-400 uppercase mt-0.5">
                             {panel.full_area_sqm.toFixed(1)}m² Effective Area
                           </p>
                         </div>
                       </div>
                       <div className="text-right">
                         <p className="text-sm font-black text-green-600 mb-2">{(panel.confidence * 100).toFixed(1)}%</p>
                         <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden border border-white shadow-inner">
                           <div 
                             className={`h-full ${panel.confidence > 0.7 ? 'bg-green-500' : panel.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`}
                             style={{ width: `${panel.confidence * 100}%` }} 
                           />
                         </div>
                       </div>
                     </div>
                   ))
                 ) : (
                   <div className="text-center py-12">
                     <FiAlertCircle className="w-16 h-16 text-gray-200 mx-auto mb-4" />
                     <p className="text-base font-black text-gray-400 uppercase tracking-widest">Scanning Anomaly</p>
                     <p className="text-xs text-gray-400 mt-2">Zero solar installations verified in this specific viewport.</p>
                   </div>
                 )}
               </div>
             </div>

             {/* Existing JSON Feature (Polished) */}
             <div className="space-y-6">
                <div className="bg-black rounded-[2.5rem] p-8 h-full shadow-2xl relative overflow-hidden group">
                  <div className="absolute top-0 right-0 w-48 h-48 bg-green-500/5 rounded-full -mr-24 -mt-24" />
                  <div className="relative h-full flex flex-col">
                    <div className="flex justify-between items-center mb-8">
                       <div className="flex items-center gap-3 text-xs font-black text-green-500/50 uppercase tracking-[0.2em]">
                         <FiCode className="text-xl" /> Development Metadata
                       </div>
                       <button 
                         onClick={() => setShowJson(!showJson)}
                         className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-xl text-[10px] font-black transition-all border border-white/10 uppercase tracking-widest"
                       >
                         {showJson ? 'Minimize' : 'View Raw'}
                       </button>
                    </div>
                    
                    <div className="flex-grow">
                      <AnimatePresence mode="wait">
                        {showJson ? (
                          <motion.div
                            key="json-view"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="h-full max-h-[400px] overflow-auto custom-scrollbar-dark"
                          >
                            <pre className="text-green-400/80 text-[10px] font-mono leading-relaxed">
                              {JSON.stringify(data, null, 2)}
                            </pre>
                          </motion.div>
                        ) : (
                          <motion.div
                            key="placeholder"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="flex flex-col items-center justify-center h-full text-center py-12 opacity-30"
                          >
                            <FiSearch className="text-6xl text-white mb-6" />
                            <p className="text-xs font-black text-white uppercase tracking-[0.3em]">Encrypted System Payload</p>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    <div className="pt-8 border-t border-white/5 mt-auto">
                       <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Coordinates</p>
                            <p className="text-xs font-bold text-white mt-1">{result.lat?.toFixed(6) || 'N/A'}, {result.lng?.toFixed(6) || 'N/A'}</p>
                          </div>
                          <div>
                            <p className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Source Engine</p>
                            <p className="text-xs font-bold text-white mt-1">YOLOv8 Solar v2.4</p>
                          </div>
                       </div>
                    </div>
                  </div>
                </div>
             </div>
          </div>
          
          {/* FINAL CTA FOOTER REMOVED AS REQUESTED */}

        </div>
      </div>
    </section>
  );
};

export default ResultsSection;
