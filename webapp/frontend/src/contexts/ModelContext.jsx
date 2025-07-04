import React, { createContext, useState, useEffect, useContext } from "react";
import { api } from "../services/api.js";
const ModelContext = createContext();
export const ModelProvider = ({
  children
}) => {
  const [modelDetails, setModelDetails] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    const fetchModelDetails = async () => {
      try {
        const details = await api.getModelDetails();
        setModelDetails(details);
        setIsLoading(false);
      } catch (err) {
        console.error("Failed to fetch model details:", err);
        setError("Failed to load model information. Please try again later.");
        setIsLoading(false);
      }
    };
    fetchModelDetails();
  }, []);
  return <ModelContext.Provider value={{
    modelDetails,
    isLoading,
    error
  }}>
    {children}
  </ModelContext.Provider>;
};
export const useModelDetails = () => useContext(ModelContext);