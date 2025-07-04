import React from 'react';
import { BarChart3Icon, ActivityIcon } from 'lucide-react';
const ResultsDisplay = ({
  results
}) => {
  // Determine color based on stage
  const getStageColor = (class_code) => {
    switch (class_code) {
      case 'CN':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300';
      case 'EMCI':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300';
      case 'LMCI':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300';
      case 'AD':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
    }
  };
  const getConfidenceBarColor = (confidence) => {
    if (confidence >= 80) {
      return 'bg-green-500';
    } else if (confidence >= 60) {
      return 'bg-yellow-500';
    } else {
      return 'bg-red-500';
    }
  };
  // Convert decimal to percentage
  const toPercentage = value => {
    return `${Math.round(value * 100)}%`;
  };
  return <div className="space-y-8">
    {/* Classification Result */}
    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 border border-gray-100 dark:border-gray-600">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">
        Classification Result
      </h3>
      <div className="flex flex-col md:flex-row md:items-center justify-between">
        <div className="mb-4 md:mb-0">
          <span className="text-sm text-gray-500 dark:text-gray-300 block mb-1">
            Detected Stage:
          </span>
          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getStageColor(results.stageCode)}`}>
            {results.stage}
          </span>
        </div>
        <div>
          <span className="text-sm text-gray-500 dark:text-gray-300 block mb-1">
            Confidence Score:
          </span>
          <div className="flex items-center">
            <div className="w-32 bg-gray-200 dark:bg-gray-600 rounded-full h-2.5 mr-2">
              <div
                className={`h-full rounded-full transition-all duration-300 ${getConfidenceBarColor(results.confidence)}`}
                style={{ width: `${results.confidence}%` }}
              ></div>
            </div>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
              {`${results.confidence}%`}
            </span>
          </div>
        </div>
      </div>
    </div>
    {/* Metrics */}
    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 border border-gray-100 dark:border-gray-600">
      <div className="flex items-center mb-4">
        <BarChart3Icon className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
          Model Accuracy Metrics
        </h3>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-100 dark:border-gray-600">
          <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">Accuracy</span>
          <span className="text-lg font-semibold text-gray-800 dark:text-white">
            {`${results.metrics.accuracy}%`}
          </span>
        </div>
        <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-100 dark:border-gray-600">
          <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">Precision</span>
          <span className="text-lg font-semibold text-gray-800 dark:text-white">
            {`${results.metrics.precision}%`}
          </span>
        </div>
        <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-100 dark:border-gray-600">
          <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">Recall</span>
          <span className="text-lg font-semibold text-gray-800 dark:text-white">
            {`${results.metrics.recall}%`}
          </span>
        </div>
        <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-100 dark:border-gray-600">
          <span className="text-xs text-gray-500 dark:text-gray-400 block mb-1">F1 Score</span>
          <span className="text-lg font-semibold text-gray-800 dark:text-white">
            {`${results.metrics.f1_score}%`}
          </span>
        </div>
      </div>
    </div>
    {/* GradCAM Visualization */}
    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 border border-gray-100 dark:border-gray-600">
      <div className="flex items-center mb-4">
        <ActivityIcon className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">
          GradCAM Visualization
        </h3>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
        The highlighted areas show regions of the brain that most influenced
        the classification decision.
      </p>
      <div className="bg-black rounded-lg overflow-hidden max-w-sm mx-auto shadow-md">
        <img src={`data:image/png;base64,${results.gradcamUrl}`} alt="GradCAM visualization of brain scan" className="w-full h-auto" />
      </div>
      <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
        <p>
          Red areas indicate regions with the strongest influence on the
          classification.
        </p>
      </div>
    </div>
  </div>;
};
export default ResultsDisplay;