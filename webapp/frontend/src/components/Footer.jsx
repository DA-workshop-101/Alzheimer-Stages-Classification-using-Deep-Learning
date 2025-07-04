import React from 'react';
import { Link } from 'react-router-dom';
import { BrainIcon, GithubIcon, TwitterIcon, LinkedinIcon } from 'lucide-react';
const Footer = () => {
  return <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 py-8 transition-colors duration-200">
      <div className="container mx-auto px-4">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <BrainIcon className="h-6 w-6 text-blue-600 dark:text-blue-400 mr-2" />
            <span className="text-gray-700 dark:text-gray-300 font-medium">
              ADAPT - Alzheimer Disease Analysis and Prediction Tool
            </span>
          </div>
          <div className="flex space-x-6">
            <a href="https://github.com/DA-workshop-101/Alzheimer-Stages-Classification-using-Deep-Learning" target='_blank' className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" aria-label="GitHub">
              <GithubIcon className="h-5 w-5" />
            </a>
            <a href="https://x.com/tabish_ali004" target='_blank' className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" aria-label="Twitter">
              <TwitterIcon className="h-5 w-5" />
            </a>
            <a href="https://www.linkedin.com/in/sushantsunilshinde/" target='_blank' className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" aria-label="LinkedIn">
              <LinkedinIcon className="h-5 w-5" />
            </a>
          </div>
        </div>
        <div className="mt-8 border-t border-gray-100 dark:border-gray-700 pt-6 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>
            &copy; {new Date().getFullYear()} ADAPT Research Team. All rights
            reserved.
          </p>
          <div className="mt-2 flex justify-center space-x-4">
            <Link to="/" className="hover:text-blue-600 dark:hover:text-blue-400">
              Privacy Policy
            </Link>
            <Link to="/" className="hover:text-blue-600 dark:hover:text-blue-400">
              Terms of Use
            </Link>
            <Link to="/" className="hover:text-blue-600 dark:hover:text-blue-400">
              Contact
            </Link>
          </div>
        </div>
      </div>
    </footer>;
};
export default Footer;