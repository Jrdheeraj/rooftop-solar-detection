// API service functions for backend communication

const API_BASE_URL = '/api';

export const apiService = {
  // Health check
  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Predict solar panels on uploaded image
  async predictImage(file, confidence = 0.5) {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('confidence', confidence.toString());

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  },

  // Predict using coordinates (placeholder)
  async predictByCoords(lat, lng, confidence = 0.5) {
    try {
      const params = new URLSearchParams({
        lat: lat.toString(),
        lng: lng.toString(),
        confidence: confidence.toString()
      });

      const response = await fetch(`${API_BASE_URL}/coords?${params}`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Coordinate prediction failed:', error);
      throw error;
    }
  },

  // Batch prediction for multiple images
  async batchPredict(files, confidence = 0.5) {
    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('confidence', confidence.toString());

      const response = await fetch(`${API_BASE_URL}/batch`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Batch prediction failed:', error);
      throw error;
    }
  }
};
