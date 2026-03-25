import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import HowItWorks from './components/HowItWorks';
import InputSection from './components/InputSection';
import Loader from './components/Loader';
import ResultsSection from './components/ResultsSection';
import WhyChooseUs from './components/WhyChooseUs';
import UseCases from './components/UseCases';
import CTASection from './components/CTASection';
import Footer from './components/Footer';
import './index.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = (data) => {
    setLoading(true);
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setResult({
        overlay_path: "https://images.unsplash.com/photo-1509395176047-4a66953fd231?auto=format&fit=crop&w=1600&q=80",
        has_solar: true,
        confidence: 0.947,
        panels_in_buffer: 12,
        total_area: 42.8,
        lat: data.type === 'coords' ? data.data.lat : 17.4482,
        lng: data.type === 'coords' ? data.data.lng : 78.3915,
        panels: [
          { id: 1, confidence: 0.98, area: 2.1 },
          { id: 2, confidence: 0.97, area: 2.2 },
          { id: 3, confidence: 0.94, area: 2.0 },
          { id: 4, confidence: 0.99, area: 2.3 },
          { id: 5, confidence: 0.92, area: 2.1 },
          { id: 6, confidence: 0.88, area: 1.9 },
        ]
      });
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }, 4500);
  };

  const handleReset = () => {
    setResult(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="bg-white min-h-screen">
      <Navbar />
      <div id="home">
        <Hero />
      </div>
      <div id="how-it-works">
        <HowItWorks />
      </div>
      
      {!result && (
        <div id="analyze">
          <InputSection onAnalyze={handleAnalyze} />
        </div>
      )}
      
      {loading && <Loader />}
      
      {result && (
        <div id="results">
          <ResultsSection data={result} onReset={handleReset} />
        </div>
      )}

      {!result && (
        <>
          <div id="features">
            <WhyChooseUs />
          </div>
          <div id="solutions">
            <UseCases />
          </div>
        </>
      )}

      <CTASection />
      <div id="about">
        <Footer />
      </div>
    </div>
  );
}

export default App;
