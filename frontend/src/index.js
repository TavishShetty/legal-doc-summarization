import React from 'react';
import { createRoot } from 'react-dom/client';  // Updated import
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';  // Optional, remove if index.css isn’t needed yet

const root = createRoot(document.getElementById('root'));  // Create root
root.render(
  <BrowserRouter>
    <App />
  </BrowserRouter>
);