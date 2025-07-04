import React, { useState, useRef } from 'react';
import { UploadCloudIcon, FileIcon, XIcon } from 'lucide-react';
const FileUpload = ({
  onFileUpload,
  isProcessing,
  onFileClear
}) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);
  const handleFileChange = e => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      // Create preview URL for image
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  const handleDragOver = e => {
    e.preventDefault();
    e.stopPropagation();
  };
  const handleDrop = e => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      // Create preview URL for image
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };
  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    onFileClear?.();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  const handleUpload = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    }
  };
  return <div className="w-full">
    {!selectedFile ? <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors" onDragOver={handleDragOver} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()}>
      <UploadCloudIcon className="h-12 w-12 text-gray-400 dark:text-gray-500 mx-auto mb-4" />
      <h3 className="text-lg font-medium text-gray-700 dark:text-gray-200 mb-1">
        Upload Brain Scan
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Drag and drop or click to select
      </p>
      <p className="text-xs text-gray-400 dark:text-gray-500">
        Supported formats: JPEG, PNG, DICOM
      </p>
      <input type="file" className="hidden" accept="image/*" ref={fileInputRef} onChange={handleFileChange} />
    </div> : <div className="border rounded-lg p-4 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <FileIcon className="h-5 w-5 text-blue-500 dark:text-blue-400 mr-2" />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-200 truncate" style={{
            maxWidth: '200px'
          }}>
            {selectedFile.name}
          </span>
        </div>
        <button onClick={handleRemoveFile} className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300" disabled={isProcessing}>
          <XIcon className="h-5 w-5" />
        </button>
      </div>
      {previewUrl && <div className="mb-4">
        <div className="w-full bg-black rounded-md overflow-hidden max-w-sm mx-auto" style={{
          maxHeight: '400px'
        }}>
          <img src={previewUrl} alt="Brain scan preview" className="w-full h-full object-contain" />
        </div>
      </div>}
      <button onClick={handleUpload} disabled={isProcessing} className={`w-full py-2 px-4 rounded-md text-white font-medium flex items-center justify-center ${isProcessing ? 'bg-blue-400 dark:bg-blue-500 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'}`}>
        {isProcessing ? <>
          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Processing...
        </> : <>
          <UploadCloudIcon className="h-4 w-4 mr-2" />
          Analyze Brain Scan
        </>}
      </button>
    </div>}
  </div>;
};
export default FileUpload;