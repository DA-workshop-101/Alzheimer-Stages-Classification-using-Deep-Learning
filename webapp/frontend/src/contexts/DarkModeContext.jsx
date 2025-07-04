import React, { createContext, useState, useContext, useEffect } from 'react';
const DarkModeContext = createContext();
export const DarkModeProvider = ({
  children
}) => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check if user has previously set a preference
    const savedMode = localStorage.getItem('darkMode');
    return savedMode === 'true' ? true : false;
  });
  useEffect(() => {
    // Update localStorage when mode changes
    localStorage.setItem('darkMode', isDarkMode);
    // Update document class for global styling
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);
  
  const toggleDarkMode = () => {
    setIsDarkMode(prev => !prev);
  };

  const value = { isDarkMode, toggleDarkMode };

  return (
    <DarkModeContext.Provider value={value}>
      {children}
    </DarkModeContext.Provider>
  );
};
export const useDarkMode = () => useContext(DarkModeContext);