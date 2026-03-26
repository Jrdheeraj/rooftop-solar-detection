// API service functions for backend communication
const API_URL = import.meta.env.VITE_API_URL || "https://rooftop-solar-detection.onrender.com";

export const apiService = {
  // Predict solar panels on uploaded image
  async analyzeImage(file, confidence = 0.5) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Send confidence as a query parameter as required by the backend
      const response = await fetch(`${API_URL}/predict?confidence=${confidence}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Analysis failed:', error);
      throw error;
    }
  },

  // Health check
  async healthCheck() {
    try {
      const response = await fetch(`${API_URL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
};
