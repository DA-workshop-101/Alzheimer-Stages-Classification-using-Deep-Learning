const API_BASE_URL = import.meta.env.VITE_BACKEND_URL; // Set to empty string for relative URLs or specify your API base URL here
export const api = {
  ping: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ping`);
      return response.ok;
    } catch (error) {
      console.error("Health check failed:", error);
      return false;
    }
  },
  getModelDetails: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model-details`, {
        method: "GET"
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch model details: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Failed to get model details:", error);
      throw error;
    }
  },
  predict: async imageFile => {
    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Prediction failed:", error);
      throw error;
    }
  }
};