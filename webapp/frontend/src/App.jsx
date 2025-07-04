import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import ClassificationPage from './pages/ClassificationPage';
import { DarkModeProvider } from './contexts/DarkModeContext.jsx';
import { ModelProvider } from './contexts/ModelContext.jsx'
export default function App() {
  return <DarkModeProvider>
    <ModelProvider>
      <BrowserRouter>
        <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
          <Header />
          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/classification" element={<ClassificationPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </BrowserRouter>
    </ModelProvider>
  </DarkModeProvider>;
}