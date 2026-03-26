// API service functions for backend communication
import { resolveApiBaseUrl } from './apiConfig';

const API_URL = resolveApiBaseUrl(import.meta.env);

export const apiService = {
  // 1. IMAGE ANALYSIS (Multipart Form Data)
  async analyzeImage(file, confidence = 0.5, options = {}) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("image_type", options.imageType || "PHOTO");
    formData.append("buffer_radius_sqft", String(options.buffer ?? 0));

    if (options.latitude !== undefined && options.latitude !== null && options.latitude !== "") {
      formData.append("latitude", String(options.latitude));
    }
    if (options.longitude !== undefined && options.longitude !== null && options.longitude !== "") {
      formData.append("longitude", String(options.longitude));
    }

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
  async analyzeCoordinates(lat, lon, confidence = 0.5, buffer = 1200) {
    const response = await fetch(`${API_URL}/coords?confidence=${confidence}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        latitude: lat,
        longitude: lon,
        buffer_radius_sqft: Number(buffer)
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
