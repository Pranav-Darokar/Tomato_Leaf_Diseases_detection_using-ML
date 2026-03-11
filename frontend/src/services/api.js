// Real API integration with FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

/**
 * Sends a tomato leaf image to the FastAPI /predict endpoint
 * and returns structured disease detection results.
 */
export const analyzeImage = async (file) => {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            let errMsg = 'Server error. Please try again.';
            try {
                const errData = await response.json();
                errMsg = errData.detail || errMsg;
            } catch { /* ignore parse error */ }
            return { success: false, error: errMsg };
        }

        const result = await response.json();

        if (!result.success) {
            return {
                success: false,
                error: result.error || 'No disease detected. Please upload a clear tomato leaf image.',
            };
        }

        return {
            success: true,
            data: {
                isHealthy: result.isHealthy,
                diseases: result.diseases,
            },
        };
    } catch (error) {
        console.error('API call error:', error);
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            return {
                success: false,
                error: 'Cannot connect to the detection server. Please make sure the backend is running on port 8000.',
            };
        }
        return {
            success: false,
            error: 'Failed to analyze the image. Please try again with a clear tomato leaf photo.',
        };
    }
};
