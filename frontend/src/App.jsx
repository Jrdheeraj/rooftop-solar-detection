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
  const [result, setResult] = useState(null);

  const handleAnalyze = async (data) => {
    setLoading(true);
    try {
      let response;
      
      if (data.type === 'upload') {
        response = await apiService.predictImage(data.data, data.confidence);
      } else if (data.type === 'coords') {
        response = await apiService.predictByCoords(
          parseFloat(data.data.lat), 
          parseFloat(data.data.lng), 
          data.confidence
        );
      }

      if (response.status === 'success') {
        // Use the overlay image with bounding boxes from the backend
        // This image contains the actual bounding boxes and annotations
        let overlayImage = null;
        if (response.overlay_image) {
          // Use the generated overlay with bounding boxes
          overlayImage = response.overlay_image;
        } else if (data.type === 'coords' && response.satellite_image_url) {
          // Fallback to satellite image for coordinate analysis
          overlayImage = response.satellite_image_url;
        } else if (data.type === 'upload') {
          // Fallback to uploaded image
          overlayImage = URL.createObjectURL(data.data);
        } else {
          // Final fallback
          const lat = parseFloat(data.data?.lat) || 0;
          const lng = parseFloat(data.data?.lng) || 0;
          overlayImage = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lng}&zoom=19&size=400x400&maptype=satellite&key=YOUR_API_KEY`;
        }

        // Transform API response to match the expected JSON format
        const transformedResult = {
          sample_id: response.sample_id || 9999,
          latitude: response.latitude || response.coordinates?.lat || parseFloat(data.data?.lat) || 0,
          longitude: response.longitude || response.coordinates?.lng || parseFloat(data.data?.lng) || 0,
          has_solar: response.has_solar || false,
          confidence: response.confidence || 0,
          pv_area_sqm_est: response.pv_area_sqm_est || 0,
          buffer_radius_sqft: response.buffer_radius_sqft || data.buffer || 1200,
          panels_in_buffer: response.panels_in_buffer || [],
          best_panel_id: response.best_panel_id || -1,
          qc_status: response.qc_status || 'NOT_VERIFIABLE',
          bbox_or_mask: response.bbox_or_mask || '',
          image_metadata: {
            source: response.image_metadata?.source || (data.type === 'coords' ? 'GOOGLE_STATIC_MAPS' : 'USER_UPLOAD'),
            capture_date: response.image_metadata?.capture_date || new Date().toISOString().split('T')[0],
            zoom: response.image_metadata?.zoom || 19,
            conf_threshold: data.confidence,
            overlap_threshold: 0.1,
            img_shape: response.image_metadata?.img_shape || [400, 400],
            qc_reasons: response.image_metadata?.qc_reasons || ['Analysis complete']
          },
          overlay_path: overlayImage
        };
        
        setResult(transformedResult);
        
        // Scroll to results
        setTimeout(() => {
          document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      } else {
        throw new Error(response.message || 'Analysis failed');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert(`Analysis failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
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
