import React from 'react';
import { Link } from 'react-router-dom';
import { BrainIcon, ActivityIcon, ClipboardCheckIcon, ArrowRightIcon } from 'lucide-react';
const HomePage = () => {
  return <div className="w-full">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-blue-900 via-blue-800 to-blue-900/90 text-white py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">ADAPT</h1>
            <p className="text-xl md:text-2xl mb-8">
              Alzheimer Disease Analysis and Prediction Tool
            </p>
            <p className="text-lg mb-8">
              Advanced AI-powered classification for early detection and
              accurate staging of Alzheimer's disease
            </p>
            <Link to="/classification" className="inline-flex items-center bg-white text-blue-700 font-medium px-6 py-3 rounded-lg hover:bg-blue-50 transition-colors">
              Try Classification Tool
              <ArrowRightIcon className="ml-2 h-5 w-5" />
            </Link>
          </div>
        </div>
      </section>
      {/* About Section */}
      <section className="relative">
        <div className="absolute inset-0 h-[600px] overflow-hidden">
          <div className="absolute inset-0 bg-blue-900/90 -translate-y-16 h-32 blur-2xl"></div>
          <img src="https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?auto=format&fit=crop&q=80" alt="Medical research environment" className="w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-b from-blue-900/95 via-blue-900/80 to-white dark:to-gray-900"></div>
        </div>
        <div className="container mx-auto px-4 relative pt-16 pb-24">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12 text-white">
              About ADAPT
            </h2>
            <div className="bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm rounded-xl shadow-xl p-8">
              <div className="prose lg:prose-lg mx-auto text-gray-600 dark:text-gray-300">
                <p>
                  ADAPT (Alzheimer Disease Analysis and Prediction Tool) is a
                  cutting-edge AI system designed to assist healthcare
                  professionals in the early detection and accurate staging of
                  Alzheimer's disease.
                </p>
                <p>
                  Using advanced deep learning algorithms trained on thousands
                  of brain scans, ADAPT can classify brain images into different
                  stages of Alzheimer's disease with high accuracy, providing
                  valuable diagnostic support for clinicians.
                </p>
                <p>
                  Our tool aims to improve early intervention, treatment
                  planning, and patient outcomes through precise classification
                  and intuitive visualization of results.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
      {/* Features Section */}
      <section className="py-16 bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12 text-gray-800 dark:text-white">
            Key Features
          </h2>
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <div className="bg-blue-100 dark:bg-blue-900 w-12 h-12 rounded-full flex items-center justify-center mb-4">
                <BrainIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800 dark:text-white">
                Advanced Classification
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Accurately classifies brain scans into different stages of
                Alzheimer's disease using state-of-the-art deep learning models.
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <div className="bg-blue-100 dark:bg-blue-900 w-12 h-12 rounded-full flex items-center justify-center mb-4">
                <ActivityIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800 dark:text-white">
                Detailed Metrics
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Provides comprehensive accuracy metrics and confidence scores to
                support clinical decision-making.
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
              <div className="bg-blue-100 dark:bg-blue-900 w-12 h-12 rounded-full flex items-center justify-center mb-4">
                <ClipboardCheckIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-800 dark:text-white">
                Visual Explanations
              </h3>
              <p className="text-gray-600 dark:text-gray-300">
                Utilizes GradCAM technology to highlight regions of interest in
                brain scans that influenced the classification decision.
              </p>
            </div>
          </div>
        </div>
      </section>
      {/* How It Works Section */}
      <section className="py-16 bg-white dark:bg-gray-800 transition-colors duration-200">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-12 text-gray-800 dark:text-white">
              How It Works
            </h2>
            <div className="space-y-8">
              <div className="flex">
                <div className="flex-shrink-0 mr-4">
                  <div className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
                    1
                  </div>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
                    Upload Brain Scan
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Upload MRI or CT scan images of the patient's brain. The
                    system accepts standard medical imaging formats.
                  </p>
                </div>
              </div>
              <div className="flex">
                <div className="flex-shrink-0 mr-4">
                  <div className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
                    2
                  </div>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
                    AI Analysis
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Our deep learning model processes the image, analyzing
                    patterns and features associated with different stages of
                    Alzheimer's disease.
                  </p>
                </div>
              </div>
              <div className="flex">
                <div className="flex-shrink-0 mr-4">
                  <div className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
                    3
                  </div>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-gray-800 dark:text-white">
                    Results & Visualization
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300">
                    Receive detailed classification results with accuracy
                    metrics and GradCAM visualizations highlighting regions of
                    interest.
                  </p>
                </div>
              </div>
            </div>
            <div className="mt-12 text-center">
              <Link to="/classification" className="inline-flex items-center bg-blue-600 text-white font-medium px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
                Try the Classification Tool
                <ArrowRightIcon className="ml-2 h-5 w-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>;
};
export default HomePage;