import React, { useState, useEffect } from "react";
import {
  HashRouter as Router,
  Route,
  Routes,
  Navigate
} from "react-router-dom";

import LandingPage from "./pages/LandingPage/landingPage";
import DetailsPage from "./pages/DetailsPage/details";
import HomePage from "./pages/HomePage/homePage"

import "./App.css";

function App() {
  return (
    <Router>
        <Routes>
          <Route path="/" element={<HomePage/>} />
          <Route path="/landing" element={<LandingPage />} />
          <Route path="/details" element={<DetailsPage />} />
        </Routes>
    </Router>
  );
}

export default App;