import React, { useState } from 'react';
import { jsPDF } from 'jspdf';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiMapPin, FiCpu, FiGrid, FiTarget, FiDownload, 
  FiCode, FiArrowLeft, FiActivity, FiSearch, 
  FiCornerDownRight, FiCheckCircle, FiAlertCircle, FiZap,
  FiDollarSign, FiTrendingUp, FiWind, FiSun, FiLayers, FiFileText,
  FiLayout
} from 'react-icons/fi';

const ResultsSection = ({ data, onReset }) => {
  // Transform the new API response format to match our UI needs
  const formatNullable = (value, digits = 2, suffix = '') => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return 'N/A';
    }
    return `${Number(value).toFixed(digits)}${suffix}`;
  };

  const result = {
    overlay_path: data?.overlay_path || data?.overlay_image || "https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
    has_solar: data?.has_solar || false,
    confidence: data?.confidence ?? 0,
    panels_in_buffer: data?.panels_in_buffer?.length || 0,
    total_area: data?.pv_area_sqm_est ?? 0,
    estimated_capacity_kw: data?.estimated_capacity_kw ?? 0,
    estimated_annual_production_kwh: data?.estimated_annual_production_kwh ?? 0,
    lat: data?.latitude ?? data?.lat ?? null,
    lng: data?.longitude ?? data?.lng ?? null,
    panels: data?.panels_in_buffer?.map((panel, idx) => ({
      id: panel.panel_id ?? idx,
      confidence: panel.conf ?? 0,
      area: panel.inside_area_sqm || panel.full_area_sqm || 0,
      full_area_sqm: panel.full_area_sqm ?? 0,
      inside_area_sqm: panel.inside_area_sqm ?? 0,
      overlap_ratio: panel.overlap_ratio ?? 0,
      bbox_center: panel.bbox_center || [],
      estimated_capacity_kw: panel.estimated_capacity_kw,
      estimated_annual_production_kwh: panel.estimated_annual_production_kwh,
      efficiency_rating: panel.efficiency_rating ?? 100,
      lifetime_validity_years: panel.lifetime_validity_years ?? null
    })) || [],
    qc_status: data?.qc_status || 'NOT_VERIFIABLE',
    buffer_radius_sqft: data?.buffer_radius_sqft || 1200,
    best_panel_id: data?.best_panel_id ?? -1,
    bbox_or_mask: data?.bbox_or_mask || '',
    image_metadata: data?.image_metadata || {},
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

  const PANEL_KW_PER_SQM = 0.2;
  const SUN_HOURS_PER_YEAR = 1460;
  const bestPanel = result.panels.find(p => p.id === result.best_panel_id) || result.panels[0] || null;
  const panelMetrics = result.panels.map((panel, index) => {
    const fallbackCapacityKw =
      panel.area > 0 ? Number((panel.area * PANEL_KW_PER_SQM).toFixed(2)) : null;
    const fallbackEnergyKwh =
      fallbackCapacityKw !== null
        ? Number((fallbackCapacityKw * SUN_HOURS_PER_YEAR).toFixed(2))
        : null;

    return {
      ...panel,
      panelNumber: index + 1,
      estimated_capacity_kw: panel.estimated_capacity_kw ?? fallbackCapacityKw,
      estimated_annual_production_kwh:
        panel.estimated_annual_production_kwh ?? fallbackEnergyKwh,
      lifetimeValidity:
        panel.lifetime_validity_years
          ? `${panel.lifetime_validity_years} Years`
          : (result.financial && 'lifetime_savings_25yr' in result.financial ? '25 Years' : 'N/A')
    };
  });

  const handleDownloadReport = () => {
    const doc = new jsPDF();
    const primaryColor = '#00c96a';
    doc.setFontSize(22);
    doc.setTextColor(primaryColor);
    doc.text('SolarScan AI - Intelligence Report', 20, 20);
    doc.setFontSize(10);
    doc.setTextColor(150, 150, 150);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 28);
    doc.setDrawColor(230, 230, 230);
    doc.line(20, 35, 190, 35);
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.text('Analysis Metrics:', 20, 50);
    const stats = [
      ['Total Panels', `${result.panels.length} Units`],
      ['Total PV Area', `${result.total_area.toFixed(1)} m²`],
      ['Peak Capacity', `${result.estimated_capacity_kw.toFixed(2)} kW`],
      ['Annual Production', `${Math.round(result.estimated_annual_production_kwh).toLocaleString()} kWh/y`],
      ['Model Confidence', `${(result.confidence * 100).toFixed(1)}%`],
      ['QC Status', result.qc_status]
    ];
    stats.forEach((stat, i) => {
      doc.setFontSize(11);
      doc.text(stat[0], 25, 65 + (i * 12));
      doc.text(stat[1], 100, 65 + (i * 12));
    });
    doc.save(`SolarScan_Report_${result.lat || '0'}_${result.lng || '0'}.pdf`);
  };

  return (
    <>
      <style>{CSS}</style>
      <div className="max-w-[1240px] mx-auto px-6 py-12">
        <div className="flex justify-between items-center mb-8">
           <button onClick={onReset} className="flex items-center gap-2 text-xs font-black text-gray-500 hover:text-green-500 uppercase tracking-[0.2em] transition-colors group">
             <FiArrowLeft className="group-hover:-translate-x-1 transition-transform" /> BACK TO SCAN
           </button>
           <div className="flex gap-4">
              <button onClick={handleDownloadReport} className="flex items-center gap-3 bg-white text-gray-900 border border-gray-200 px-6 py-2.5 rounded-full text-[10px] font-black uppercase tracking-widest hover:border-gray-900 transition-all shadow-sm">
                 <FiDownload /> Export PDF
              </button>
              <button className="flex items-center gap-3 bg-green-600 text-white px-6 py-2.5 rounded-full text-[10px] font-black uppercase tracking-widest hover:bg-green-700 transition-all shadow-lg shadow-green-500/20">
                 <FiZap /> Consult Specialist
              </button>
           </div>
        </div>

        <div className="ss-box" style={s.darkBox}>
          <div style={s.pillRow}>
            <div className="ss-pill-dark" style={s.pillDark}>
              <Icon n="search" size={12} color="#8899aa" />
              <span>AI DETECTION ENGINE</span>
            </div>
            <div className="ss-pill-green" style={s.pillGreen}>
              <Icon n="sun" size={12} color="#fff" />
              <span>{result.panels.length} PANELS LOCATED</span>
            </div>
            <div className="ss-pill-blue" style={s.pillBlue}>
              <Icon n="check" size={12} color="#fff" />
              <span>STATUS: {result.qc_status}</span>
            </div>
          </div>

          <div className="ss-det-title" style={s.titleRow}>
            <span style={s.titleDot} />
            <span style={s.titleText}>Solar Intelligence Report — {result.panels.length > 0 ? `${result.panels.length} PANELS DETECTED` : 'SCANNING FOR TARGETS'}</span>
            <span style={s.titleDot} />
          </div>

          <div style={s.grid}>
            <div className="ss-left-card" style={s.sideCard}>
              <Block title="VISUAL LEGEND" icon="layers" ic="#00c96a">
                <div className="ss-legend-item" style={s.legendItem}>
                  <span style={{ ...s.dot, background:"#00ff88", boxShadow:"0 0 8px #00ff8899" }} />
                  <span style={s.legendTxt}>Lime: Prime Target</span>
                </div>
                <div className="ss-legend-item" style={s.legendItem}>
                  <span style={{ ...s.dot, background:"#00e5ff", boxShadow:"0 0 8px #00e5ff99" }} />
                  <span style={s.legendTxt}>Cyan: Standard Panel</span>
                </div>
              </Block>
              <div style={s.div} />
              <Block title="DETECTION SPECS" icon="cpu" ic="#00c96a">
                <SR icon="cpu"    label="Engine" value="YOLOv8 Solar" />
                <SR icon="box"    label="Type"   value={result.image_metadata.source === 'USER_UPLOAD' ? 'PHOTO' : 'SATELLITE'} />
                <SR icon="zoom"   label="Zoom"   value="19.5x H-Res" />
                <SR icon="layers" label="Buffer" value={`${result.buffer_radius_sqft} sqft`} />
              </Block>
            </div>

            <div className="ss-center" style={s.centerCol}>
              <div style={s.imgWrap}>
                <div className="ss-scan-line" />
                <img src={result.overlay_path} alt="Solar Panel Detection" style={s.img} />
                <div style={s.gridOverlay} />
                {result.panels.slice(0, 4).map((panel, i) => (
                   <Tag 
                    key={i}
                    lbl={`P${panel.id} | ${panel.confidence.toFixed(2)} | ${panel.area.toFixed(1)}m²`} 
                    clr={panel.id === result.best_panel_id ? "#00ff88" : "#00e5ff"} 
                    top={`${15 + i*18}%`} 
                    left={`${10 + i*12}%`} 
                    del={`${1 + i*0.2}s`} 
                   />
                ))}
              </div>
            </div>

            <div className="ss-right-card" style={s.sideCard}>
              <Block title="PANEL ANALYTICS" icon="eye" ic="#00c96a">
                <MR icon="check"    label="STATUS" value={result.has_solar ? "VERIFIED" : "NONE"}   hi={result.has_solar} />
                <MR icon="layers"   label="COUNT"  value={`${result.panels.length} Units`} hi />
                <MR icon="activity" label="CONF" value={`${(result.confidence * 100).toFixed(1)}%`} />
                <MR icon="check"    label="QC"    value={result.qc_status} />
              </Block>
              <div style={s.div} />
              <Block title="BEST PANEL INFO" icon="sun" ic="#f5c518">
                <MR icon="target"   label="BEST ID" value={bestPanel ? `Panel #${bestPanel.id}` : "N/A"} hi />
                <MR icon="box"      label="MAX SIZE"     value={bestPanel ? `${bestPanel.area.toFixed(2)} m²` : "N/A"} hi />
              </Block>
              <div style={s.div} />
              <Block title="ENERGY ESTIMATES" icon="zap" ic="#f5c518">
                <MR icon="layers"   label="AREA"   value={`${result.total_area.toFixed(1)} m²`} />
                <MR icon="zap"      label="CAPACITY" value={`${result.estimated_capacity_kw.toFixed(2)} kW`}    hi />
              </Block>
            </div>
          </div>

          <div className="ss-conf-bar ss-conf-wrap" style={s.confWrap}>
            <div style={s.confLeft}>
              <Icon n="activity" size={15} color="#00c96a" />
              <span style={s.confLbl}>DETECTION CONFIDENCE</span>
            </div>
            <div style={s.confTrack}>
              <div className="ss-conf-fill" style={{...s.confFill, width: `${result.confidence * 100}%`}} />
            </div>
            <span className="ss-shimmer" style={s.confVal}>{(result.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
        
        {/* 2. KPI QUICK STATS GRID - COMPACT */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
             {[
               { icon: <FiGrid />, label: 'Total PV Area', value: `${result.total_area.toFixed(1)} sq.m`, color: 'text-gray-900' },
               { icon: <FiZap />, label: 'System Capacity', value: `${result.estimated_capacity_kw.toFixed(2)} kW`, color: 'text-blue-600' },
               { icon: <FiTrendingUp />, label: 'Annual Energy', value: `${Math.round(result.estimated_annual_production_kwh).toLocaleString()} kWh`, color: 'text-green-600' },
               { icon: <FiTrendingUp />, label: 'Detections', value: `${result.panels.length} Units`, color: 'text-purple-600' }
             ].map((stat, i) => (
               <div key={i} className="bg-white p-5 rounded-[2rem] border border-gray-100 shadow-lg flex items-center gap-4 transition-all hover:translate-y-[-2px]">
                 <div className="w-12 h-12 bg-gray-50 rounded-xl flex items-center justify-center text-xl text-gray-400">
                   {stat.icon}
                 </div>
                 <div>
                   <p className="text-xs font-black text-gray-400 uppercase tracking-widest mb-1">{stat.label}</p>
                   <p className={`text-2xl font-black ${stat.color}`}>{stat.value}</p>
                 </div>
               </div>
             ))}
          </div>

          {/* 3. MULTI-INSIGHTS GRID - COMPACT */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
            
            {/* FINANCIAL INSIGHTS */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-lg relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-32 h-32 bg-green-50 rounded-full -mr-16 -mt-16" />
              <div className="relative">
                <div className="flex items-center gap-3 text-sm font-black text-green-700 uppercase tracking-[0.2em] mb-8">
                  <FiDollarSign className="text-2xl" /> Financial
                </div>
                <div className="space-y-6">
                  <div>
                    <p className="text-xs font-black text-gray-400 uppercase tracking-widest mb-2">Investment Est.</p>
                    <p className="text-4xl font-black text-gray-900 tracking-tighter">${result.financial.est_installation_cost.toLocaleString()}</p>
                  </div>
                  <div className="pt-6 border-t border-gray-50">
                    <p className="text-xs font-black text-gray-400 uppercase tracking-widest mb-2">Lifetime Savings</p>
                    <p className="text-4xl font-black text-[#00c96a] tracking-tighter">${result.financial.lifetime_savings_25yr.toLocaleString()}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* ENVIRONMENTAL IMPACT */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-lg relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 rounded-full -mr-16 -mt-16" />
              <div className="relative">
                <div className="flex items-center gap-3 text-sm font-black text-blue-700 uppercase tracking-[0.2em] mb-8">
                  <FiWind className="text-2xl" /> Environmental
                </div>
                <div className="space-y-8">
                  <div className="flex items-center gap-5">
                    <div className="w-14 h-14 bg-blue-50 text-blue-600 rounded-xl flex items-center justify-center text-2xl">
                      <FiZap />
                    </div>
                    <div>
                      <p className="text-xs font-black text-gray-400 uppercase tracking-widest mb-1">CO2 Offset/Yr</p>
                      <p className="text-2xl font-black text-gray-900">{result.environmental.co2_saved_tons_yr.toFixed(1)} Tons</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-5">
                    <div className="w-14 h-14 bg-green-50 text-green-600 rounded-xl flex items-center justify-center text-2xl">
                      <FiSun />
                    </div>
                    <div>
                      <p className="text-xs font-black text-gray-400 uppercase tracking-widest mb-1">Trees Equivalent</p>
                      <p className="text-2xl font-black text-gray-900">{result.environmental.trees_planted_equiv} Trees</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* TECHNICAL DATA */}
            <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-lg relative overflow-hidden group">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-50 rounded-full -mr-16 -mt-16" />
              <div className="relative">
                <div className="flex items-center gap-3 text-sm font-black text-purple-700 uppercase tracking-[0.2em] mb-8">
                  <FiCpu className="text-2xl" /> Integrity
                </div>
                <div className="space-y-6">
                  <div className="flex justify-between items-center py-2.5 border-b border-gray-50">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Irradiance</span>
                    <span className="text-xl font-black text-gray-900">{result.technical.irradiance_kwh_m2_day.toFixed(1)} kWh/m²</span>
                  </div>
                  <div className="flex justify-between items-center py-2.5 border-b border-gray-50">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Inv. Spec</span>
                    <span className="text-xl font-black text-gray-900">{result.technical.recommended_inverter_kw.toFixed(1)} kW</span>
                  </div>
                  <div className="flex justify-between items-center py-2.5">
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Storage</span>
                    <span className="text-xl font-black text-blue-600">{result.technical.potential_storage_kwh.toFixed(1)} kWh</span>
                  </div>
                </div>
              </div>
            </div>

          </div>

          <div className="bg-white p-8 rounded-[2.5rem] border border-gray-100 shadow-lg mt-8 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-40 h-40 bg-green-50 rounded-full -mr-20 -mt-20" />
            <div className="relative">
              <div className="flex items-center justify-between gap-4 mb-8">
                <div className="flex items-center gap-3 text-sm font-black text-green-700 uppercase tracking-[0.2em]">
                  <FiGrid className="text-2xl" /> Panel Intelligence
                </div>
                <div className="text-xs font-black text-gray-400 uppercase tracking-widest">
                  {panelMetrics.length} Panels Shown
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                {panelMetrics.map((panel) => (
                  <div key={panel.id} className="bg-[#f8fafc] border border-gray-100 rounded-[2rem] p-6 shadow-sm">
                    <div className="flex items-start justify-between gap-3 mb-5">
                      <div>
                        <p className="text-xs font-black text-gray-400 uppercase tracking-widest mb-2">
                          Panel Number
                        </p>
                        <p className="text-2xl font-black text-gray-900">Panel #{panel.panelNumber}</p>
                      </div>
                      <div className="px-3 py-1.5 rounded-full bg-green-50 text-green-700 text-xs font-black uppercase tracking-widest">
                        {panel.confidence.toFixed(2)} Conf
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex justify-between items-center py-2.5 border-b border-gray-200/70">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Area</span>
                        <span className="text-lg font-black text-gray-900">{panel.area.toFixed(2)} m²</span>
                      </div>
                      <div className="flex justify-between items-center py-2.5 border-b border-gray-200/70">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Lifetime Validity</span>
                        <span className="text-lg font-black text-gray-900">{panel.lifetimeValidity}</span>
                      </div>
                      <div className="flex justify-between items-center py-2.5 border-b border-gray-200/70">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Capacity</span>
                        <span className="text-lg font-black text-blue-600">
                          {formatNullable(panel.estimated_capacity_kw, 2, ' kW')}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-2.5 border-b border-gray-200/70">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Efficiency Rating</span>
                        <span className="text-lg font-black text-amber-600">{panel.efficiency_rating}%</span>
                      </div>
                      <div className="flex justify-between items-center py-2.5">
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Energy Creation</span>
                        <span className="text-lg font-black text-green-600">
                          {panel.estimated_annual_production_kwh === null || panel.estimated_annual_production_kwh === undefined
                            ? 'N/A'
                            : `${Math.round(panel.estimated_annual_production_kwh).toLocaleString()} kWh/y`}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

      </div>
    </>
  );
};

/* ─── Inject keyframes once ─── */
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
  @keyframes fadeSlideUp { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:translateY(0); } }
  @keyframes fadeSlideLeft { from { opacity:0; transform:translateX(-24px); } to { opacity:1; transform:translateX(0); } }
  @keyframes fadeSlideRight { from { opacity:0; transform:translateX(24px); } to { opacity:1; transform:translateX(0); } }
  @keyframes fillBar { from { width:0%; } }
  @keyframes scanLine { 0% { top:-2px; opacity:0.8; } 100% { top:102%; opacity:0; } }
  @keyframes tagPop { 0% { opacity:0; transform:scale(0.5); } 70% { transform:scale(1.12); } 100% { opacity:1; transform:scale(1); } }
  @keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
  @keyframes borderPulse { 0%,100% { border-color: rgba(0,201,106,0.28); } 50% { border-color: rgba(0,201,106,0.75); } }
  @keyframes dotBeat { 0%,100% { transform:scale(1); } 50% { transform:scale(1.5); } }

  .ss-box * { box-sizing: border-box; }
  .ss-pill-dark  { animation: fadeSlideUp 0.5s ease both; animation-delay:0.08s; }
  .ss-pill-green { animation: fadeSlideUp 0.5s ease both; animation-delay:0.18s; }
  .ss-pill-blue  { animation: fadeSlideUp 0.5s ease both; animation-delay:0.28s; }
  .ss-det-title  { animation: fadeSlideUp 0.6s ease both; animation-delay:0.34s; }
  .ss-left-card  { animation: fadeSlideLeft 0.7s cubic-bezier(.2,.8,.4,1) both; animation-delay:0.4s; }
  .ss-right-card { animation: fadeSlideRight 0.7s cubic-bezier(.2,.8,.4,1) both; animation-delay:0.4s; }
  .ss-center     { animation: fadeSlideUp 0.65s ease both; animation-delay:0.44s; }
  .ss-conf-bar   { animation: fadeSlideUp 0.6s ease both; animation-delay:0.72s; }
  .ss-metric-row { transition: background 0.18s, transform 0.18s; border-radius:8px; padding:8px 10px; margin:0 -8px; }
  .ss-metric-row:hover { background:rgba(0,201,106,0.08); transform:translateX(3px); }
  .ss-spec-row { transition: background 0.18s, transform 0.18s; border-radius:8px; padding:6px 10px; margin:0 -8px; }
  .ss-spec-row:hover { background:rgba(0,201,106,0.07); transform:translateX(3px); }
  .ss-legend-item { transition: transform 0.2s; cursor:default; padding: 3px 0; }
  .ss-legend-item:hover { transform:translateX(4px); }
  .ss-conf-fill { animation: fillBar 1.5s cubic-bezier(.4,0,.2,1) both; animation-delay:0.85s; }
  .ss-shimmer {
    background: linear-gradient(90deg, #00c96a 0%, #00ffaa 40%, #00c96a 65%, #00c96a 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 2.4s linear infinite;
  }
  .ss-scan-line {
    position:absolute; left:0; right:0; height:2px;
    background:linear-gradient(90deg, transparent, rgba(0,201,106,0.75), transparent);
    animation: scanLine 3s ease-in-out infinite;
    pointer-events:none; z-index:4;
  }
  .ss-tag { animation: tagPop 0.45s cubic-bezier(.4,0,.2,1) both; }
  .ss-conf-wrap { animation: borderPulse 2.6s ease-in-out infinite; }
  .ss-section-icon { transition: transform 0.28s; display:flex; align-items:center; }
  .ss-section-block:hover .ss-section-icon { transform: rotate(-12deg) scale(1.2); }
`;

const Icon = ({ n, size = 15, color = "currentColor" }) => {
  const d = {
    eye: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
    zap: "M13 2 3 14h9l-1 8 10-12h-9l1-8z",
    map: "M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z M12 7a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
    cpu: "M4 4h16v16H4z M9 9h6v6H9z M9 1v3 M15 1v3 M9 20v3 M15 20v3 M20 9h3 M20 15h3 M1 9h3 M1 15h3",
    check: "M20 6 9 17l-5-5",
    search: "M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16z M21 21l-4.35-4.35",
    layers: "M12 2 2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5",
    activity: "M22 12h-4l-3 9L9 3l-3 9H2",
    globe: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M2 12h20 M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z",
    zoom: "M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16z M21 21l-4.35-4.35 M11 8v6 M8 11h6",
    box: "M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z",
    sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z M12 2v2 M12 20v2 M4.93 4.93l1.41 1.41 M17.66 17.66l1.41 1.41 M2 12h2 M20 12h2 M4.93 19.07l1.41-1.41 M17.66 6.34l1.41-1.41",
    target: "M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z M12 2v2 M12 20v2 M2 12h2 M20 12h2",
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {(d[n] || "").split(" M").map((seg, i) => (
        <path key={i} d={i === 0 ? seg : "M" + seg} />
      ))}
    </svg>
  );
};

const Block = ({ title, icon, ic, children }) => (
  <div className="ss-section-block" style={s.block}>
    <div style={s.blockHdr}>
      <span className="ss-section-icon"><Icon n={icon} size={14} color={ic} /></span>
      <span style={s.blockTitle}>{title}</span>
    </div>
    <div style={s.blockLine} />
    <div style={s.blockBody}>{children}</div>
  </div>
);

const SR = ({ icon, label, value }) => (
  <div className="ss-spec-row" style={s.specRow}>
    <div style={s.rowLeft}>
      <Icon n={icon} size={13} color="#00c96a" />
      <span style={s.specLbl}>{label}</span>
    </div>
    <span style={s.specVal}>{value}</span>
  </div>
);

const MR = ({ icon, label, value, hi }) => (
  <div className="ss-metric-row" style={s.metRow}>
    <div style={s.rowLeft}>
      <Icon n={icon} size={13} color={hi ? "#00c96a" : "#aaa"} />
      <span style={s.metLbl}>{label}</span>
    </div>
    {hi ? <span className="ss-shimmer" style={s.metHi}>{value}</span> : <span style={s.metVal}>{value}</span>}
  </div>
);

const Tag = ({ lbl, clr, top, left, del }) => (
  <div className="ss-tag" style={{
    position:"absolute", top, left, fontSize:"10px", fontWeight:"800", padding:"4px 10px", borderRadius:"6px", color: clr, background:`${clr}22`, border:`1.5px solid ${clr}`, letterSpacing:"0.04em", whiteSpace:"nowrap", pointerEvents:"none", backdropFilter:"blur(6px)", animationDelay: del, fontFamily:"'DM Sans',sans-serif", zIndex: 10,
    boxShadow: `0 4px 8px ${clr}33`
  }}>
    {lbl}
  </div>
);

const s = {
  darkBox: { background:"#000000", borderRadius:"24px", padding:"18px 32px 12px", fontFamily:"'DM Sans','Segoe UI',sans-serif", width:"100%", border:"1px solid #1c2230" },
  pillRow:{ display:"flex", gap:"10px", marginBottom:"12px", flexWrap:"wrap" },
  pillDark:{ display:"flex", alignItems:"center", gap:"8px", padding:"8px 18px", borderRadius:"30px", background:"#0d1017", color:"#9aaabb", fontSize:"11px", fontWeight:"700", letterSpacing:"0.08em", border:"1px solid #252d3d" },
  pillGreen:{ display:"flex", alignItems:"center", gap:"8px", padding:"8px 18px", borderRadius:"30px", background:"linear-gradient(135deg,#00c96a,#00a857)", color:"#fff", fontSize:"11px", fontWeight:"700", letterSpacing:"0.08em" },
  pillBlue:{ display:"flex", alignItems:"center", gap:"8px", padding:"8px 18px", borderRadius:"30px", background:"linear-gradient(135deg,#0077ff,#0055cc)", color:"#fff", fontSize:"11px", fontWeight:"700", letterSpacing:"0.08em" },
  titleRow:{ display:"flex", alignItems:"center", justifyContent:"center", gap:"14px", marginBottom:"30px" },
  titleText:{ color:"#00c96a", fontFamily:"'Syne',sans-serif", fontWeight:"700", fontSize:"14px", letterSpacing:"0.12em", textTransform:"uppercase" },
  titleDot:{ display:"inline-block", width:"6px", height:"6px", borderRadius:"50%", background:"#00c96a", animation:"dotBeat 1.8s ease-in-out infinite" },
  grid:{ display:"grid", gridTemplateColumns:"220px 1fr 220px", gap:"18px", alignItems:"start" },
  sideCard:{ background:"#ffffff", borderRadius:"18px", padding:"18px 16px", border:"1.5px solid #e4e8f0", boxShadow:"0 8px 32px rgba(0,0,0,0.12)" },
  block:{ marginBottom:"4px" }, blockHdr:{ display:"flex", alignItems:"center", gap:"8px", marginBottom:"10px" },
  blockTitle:{ color:"#111", fontFamily:"'Syne',sans-serif", fontWeight:"600", fontSize:"10px", letterSpacing:"0.16em", textTransform:"uppercase", margin:0 },
  blockLine:{ height:"1.5px", background:"linear-gradient(90deg,#00c96a44,#d8dce8,transparent)", marginBottom:"12px" },
  blockBody:{ display:"flex", flexDirection:"column", gap:"4px" },
  div:{ height:"1px", background:"#e8ecf2", margin:"14px 0" },
  legendItem:{ display:"flex", alignItems:"center", gap:"10px" }, dot:{ width:"9px", height:"9px", borderRadius:"50%", flexShrink:0 },
  legendTxt:{ color:"#222", fontSize:"13px", fontWeight:"400" },
  specRow:{ display:"flex", justifyContent:"space-between", alignItems:"center" }, rowLeft:{ display:"flex", alignItems:"center", gap:"8px" },
  specLbl:{ color:"#777", fontSize:"13px", fontWeight:"400" }, specVal:{ color:"#111", fontSize:"13px", fontWeight:"500" },
  metRow:{ display:"flex", justifyContent:"space-between", alignItems:"center" },
  metLbl:{ color:"#777", fontSize:"11px", fontWeight:"500", letterSpacing:"0.08em", textTransform:"uppercase" },
  metVal:{ color:"#111", fontSize:"14px", fontWeight:"500" }, metHi:{ fontSize:"14px", fontWeight:"600" },
  centerCol:{ borderRadius:"18px", overflow:"hidden", lineHeight:0 },
  imgWrap:{ position:"relative", borderRadius:"18px", overflow:"hidden", border:"1px solid #1e2735", boxShadow:"0 8px 40px rgba(0,201,106,0.15)" },
  img:{ width:"100%", display:"block", height:"480px", objectFit:"cover", borderRadius:"18px" },
  gridOverlay:{ position:"absolute", inset:0, backgroundImage:"linear-gradient(rgba(0,201,106,0.055) 1px,transparent 1px),linear-gradient(90deg,rgba(0,201,106,0.055) 1px,transparent 1px)", backgroundSize:"40px 40px", pointerEvents:"none", zIndex:2 },
  confWrap:{ display:"flex", alignItems:"center", gap:"14px", marginTop:"18px", background:"#07090e", borderRadius:"14px", padding:"12px 20px", border:"1.5px solid rgba(0,201,106,0.28)" },
  confLeft:{ display:"flex", alignItems:"center", gap:"10px", minWidth:"160px" },
  confLbl:{ color:"#6a7590", fontSize:"11px", fontWeight:"800", letterSpacing:"0.14em", textTransform:"uppercase" },
  confTrack:{ flex:1, height:"8px", background:"#151b28", borderRadius:"99px", overflow:"hidden" },
  confFill:{ height:"100%", background:"linear-gradient(90deg,#00c96a,#00ffaa)", borderRadius:"99px" },
  confVal:{ fontWeight:"900", fontSize:"18px", minWidth:"60px", textAlign:"right" },
};

export default ResultsSection;

