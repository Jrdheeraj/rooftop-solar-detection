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
import { apiService } from './services/api';
import './index.css';

function App() {
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async (data) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response;
      if (data.type === 'upload') {
        setLoadingMessage(
          data.imageType === 'SATELLITE'
            ? 'Uploading satellite image and analyzing with coordinates...'
            : 'Uploading image and detecting solar panels...'
        );
        response = await apiService.analyzeImage(data.data, data.confidence, {
          imageType: data.imageType,
          latitude: data.latitude,
          longitude: data.longitude,
          buffer: data.buffer,
        });
      } else if (data.type === 'coords') {
        setLoadingMessage('Fetching satellite data and analyzing coordinates...');
        response = await apiService.analyzeCoordinates(
          parseFloat(data.data.lat),
          parseFloat(data.data.lng),
          data.confidence,
          data.buffer
        );
      }

      if (response && response.status === 'success') {
        setResult(response);
        
        // Scroll to results
        setTimeout(() => {
          document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      } else {
        throw new Error(response?.message || 'Analysis failed to return a successful status.');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      const msg = err.message || 'Failed to connect to the analysis engine.';
      setError(msg);
      alert(`Analysis failed: ${msg}`);
    } finally {
      setLoading(false);
      setLoadingMessage('');
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setTimeout(() => {
      document.getElementById('analyze')?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  return (
    <div className="bg-white min-h-screen relative">
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
      
      {loading && <Loader message={loadingMessage} />}
      
      {result && (
        <div id="results">
          <ResultsSection data={result} onReset={handleReset} />
        </div>
      )}

      {error && !loading && !result && (
        <div className="max-w-4xl mx-auto px-6 pb-12 text-center">
          <div className="bg-red-50 border border-red-100 p-6 rounded-[2rem]">
            <p className="text-red-600 font-bold mb-2">Analysis Error</p>
            <p className="text-red-400 text-sm">{error}</p>
            <button 
              onClick={() => setError(null)}
              className="mt-4 px-6 py-2 bg-red-600 text-white rounded-xl text-xs font-bold"
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      <div id="features">
        <WhyChooseUs />
      </div>
      <div id="solutions">
        <UseCases />
      </div>

      <CTASection />
      <div id="about">
        <Footer />
      </div>
    </div>
  );
}

export default App;
