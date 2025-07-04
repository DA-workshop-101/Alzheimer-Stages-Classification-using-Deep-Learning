import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BrainIcon, MenuIcon, XIcon, SunIcon, MoonIcon } from 'lucide-react';
import { useDarkMode } from '../contexts/DarkModeContext';
const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();
  const {
    isDarkMode,
    toggleDarkMode
  } = useDarkMode();
  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };
  return <header className="bg-white dark:bg-gray-800 shadow-sm transition-colors duration-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-2">
            <BrainIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">ADAPT</h1>
              <p className="text-xs text-gray-600 dark:text-gray-300">
                Alzheimer Disease Analysis and Prediction Tool
              </p>
            </div>
          </Link>
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <nav className="flex space-x-8">
              <Link to="/" className={`text-sm font-medium ${location.pathname === '/' ? 'text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400'}`}>
                Home
              </Link>
              <Link to="/classification" className={`text-sm font-medium ${location.pathname === '/classification' ? 'text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400'}`}>
                Classification Tool
              </Link>
            </nav>
            {/* Dark Mode Toggle */}
            <button onClick={toggleDarkMode} className="p-2 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors" aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}>
              {isDarkMode ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
            </button>
          </div>
          {/* Mobile menu button */}
          <div className="md:hidden flex items-center space-x-4">
            {/* Dark Mode Toggle (Mobile) */}
            <button onClick={toggleDarkMode} className="p-2 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200" aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}>
              {isDarkMode ? <SunIcon className="h-4 w-4" /> : <MoonIcon className="h-4 w-4" />}
            </button>
            <button onClick={toggleMenu} className="text-gray-500 dark:text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 focus:outline-none">
              {isMenuOpen ? <XIcon className="h-6 w-6" /> : <MenuIcon className="h-6 w-6" />}
            </button>
          </div>
        </div>
        {/* Mobile Navigation */}
        {isMenuOpen && <nav className="md:hidden mt-4 space-y-3 pb-3">
            <Link to="/" className={`block text-sm font-medium ${location.pathname === '/' ? 'text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300'}`} onClick={() => setIsMenuOpen(false)}>
              Home
            </Link>
            <Link to="/classification" className={`block text-sm font-medium ${location.pathname === '/classification' ? 'text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300'}`} onClick={() => setIsMenuOpen(false)}>
              Classification Tool
            </Link>
          </nav>}
      </div>
    </header>;
};
export default Header;