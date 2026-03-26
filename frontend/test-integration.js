// Test script for frontend-backend integration
import { apiService } from './src/services/api.js';

// Test health endpoint
async function testHealth() {
    try {
        const result = await apiService.healthCheck();
        console.log('Health check:', result);
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Test with a sample image (you'll need to provide an actual image file)
async function testImageUpload() {
    // Create a dummy test file (in real usage, this would be a file input)
    const testFile = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
    
    try {
        const result = await apiService.analyzeImage(testFile, 0.5);
        console.log('Image prediction result:', result);
    } catch (error) {
        console.error('Image prediction failed:', error);
    }
}

// Test coordinates endpoint
async function testCoordinates() {
    try {
        const result = await apiService.analyzeCoordinates(17.4483, 78.3915, 0.5);
        console.log('Coordinates prediction result:', result);
    } catch (error) {
        console.error('Coordinates prediction failed:', error);
    }
}

// Run all tests
console.log('Testing API integration...');
testHealth();
testCoordinates();
// testImageUpload(); // Uncomment when you have a real image file
