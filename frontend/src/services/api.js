// API service functions for backend communication
const API_URL = import.meta.env.VITE_API_URL || "https://rooftop-solar-detection.onrender.com";

export const apiService = {
  // 1. IMAGE ANALYSIS (Multipart Form Data)
  async analyzeImage(file, confidence = 0.5) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/predict?confidence=${confidence}`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    return await response.json();
  },

  // 2. COORDINATE ANALYSIS (JSON Body)
  async analyzeCoordinates(lat, lon, confidence = 0.5) {
    const response = await fetch(`${API_URL}/coords?confidence=${confidence}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        latitude: lat,
        longitude: lon
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    return await response.json();
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
