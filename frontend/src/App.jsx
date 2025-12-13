import { useState, useEffect, useRef } from "react";
import { Routes, Route } from "react-router-dom";
import { Outlet } from "react-router-dom";

import Navbar from "./components/Navbar";
import Predict from "./components/Predict.jsx";
import Recall from "./components/Predict.jsx";

function Layout() {
  return (
    <>
      <Navbar />
      
      <Outlet />
    </>
  );
}

function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen px-4">
      <a
        className="text-4xl font-bold text-black/90 dark:text-white/90 mb-4"
        to="/"
        href="/"
      >
        AI Thoroughbred Analytics
      </a>
      <p className="text-lg text-black/60 dark:text-white/60 mb-8">
        Predict horse racing outcomes with AI
      </p>
      <a
        href="/Predict"
        className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
      >
        Get Started
      </a>
    </div>
  );
}

const App = () => {
  return (
    <div className="min-h-screen bg-zinc-100 dark:bg-neutral-900">
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/Predict" element={<Predict />} />
          <Route path="/Recall" element={<Recall />} />
        </Route>
      </Routes>
    </div>
  );
};
export default App;
