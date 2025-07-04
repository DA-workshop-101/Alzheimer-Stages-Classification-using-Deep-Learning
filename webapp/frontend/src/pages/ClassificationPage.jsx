import React, { useState } from 'react';
import { UploadCloudIcon, AlertCircleIcon, CheckCircleIcon } from 'lucide-react';
import FileUpload from '../components/FileUpload.jsx';
import ResultsDisplay from '../components/ResultsDisplay.jsx';
import { api } from "../services/api.js";
import { useModelDetails } from '../contexts/ModelContext.jsx';

const ClassificationPage = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const {
    modelDetails
  } = useModelDetails();

  const handleFileClear = () => {
    setResults(null);       // remove ResultsDisplay
    setError('');           // optional: also clear error
  };

  const handleFileUpload = async file => {
    setError('');
    setIsProcessing(true);
    setResults(null);
    try {
      // Call the predict API endpoint
      const predictionResult = await api.predict(file);
      // Format the results with the prediction data and model metrics
      const formattedResults = {
        stage: predictionResult.predicted_class,
        confidence: predictionResult.confidence || 85,
        stageCode: predictionResult.class_code,
        metrics: modelDetails || {
          accuracy: 0.85,
          precision: 0.82,
          recall: 0.79,
          f1Score: 0.81
        },
        gradcamUrl: predictionResult.gradcam || 'https://miro.medium.com/v2/resize:fit:1400/1*FgLbPRQXXPIpQMrRaG3qFg.png' // Use default if not provided
      };
      setResults(formattedResults);
    } catch (err) {
      console.error('Error during prediction:', err);
      setError("Could not process the image. Please ensure it's a valid brain scan in a supported format.");
    } finally {
      setIsProcessing(false);
    }
  };
  return <div className="w-full bg-gray-50 dark:bg-gray-900 py-12 transition-colors duration-200">
    <div className="container mx-auto px-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">
          Alzheimer's Disease Classification Tool
        </h1>
        <p className="text-gray-600 dark:text-gray-300 mb-8">
          Upload a brain scan image to classify the stage of Alzheimer's
          disease
        </p>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
          <div className="p-6">
            <FileUpload onFileUpload={handleFileUpload} isProcessing={isProcessing} onFileClear={handleFileClear} />
            {error && <div className="mt-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4 flex items-start">
              <AlertCircleIcon className="h-5 w-5 text-red-500 dark:text-red-400 mt-0.5 mr-3 flex-shrink-0" />
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>}
            {results && <div className="mt-8">
              <div className="flex items-center mb-6">
                <CheckCircleIcon className="h-5 w-5 text-green-500 dark:text-green-400 mr-2" />
                <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
                  Analysis Complete
                </h2>
              </div>
              <ResultsDisplay results={results} />
            </div>}
          </div>
        </div>
        <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg p-4 text-sm text-blue-700 dark:text-blue-300">
          <p>
            <strong>Note:</strong> This is a demonstration tool. In a
            real-world application, the classification would be performed by a
            trained AI model on a secure server. Always consult with
            healthcare professionals for medical diagnoses.
          </p>
        </div>
      </div>
    </div>
  </div>;
};
export default ClassificationPage;