"use client";

import React, { useState } from "react";
import NavigationBar from "../components/NavigationBar/NavigationBar";
import MotorHealthBar from "../components/MotorHealthBar/MotorHealthBar";

export function LearnMore() {
  const [motorHealth, setMotorHealth] = useState<number>(85);

  return (
    <div className="min-h-screen bg-black text-white">
      <NavigationBar />
      <div className="container mx-auto px-4 py-8 flex flex-col items-center">
        <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
          Learn More
        </h1>

        <div className="w-full max-w-2xl bg-gray-900 border border-gray-800 rounded-xl p-8 shadow-2xl space-y-8">
          <section className="text-center space-y-4">
            <h2 className="text-2xl font-semibold text-gray-200">
              System Diagnostics
            </h2>
            <p className="text-gray-400">
              Real-time monitoring of motor performance and structural
              integrity.
            </p>
          </section>

          {/* Motor Health Visualization */}
          <div className="py-6">
            <MotorHealthBar health={motorHealth} />
          </div>

          {/* Simulation Controls */}
          <div className="space-y-4 pt-6 border-t border-gray-800">
            <label
              htmlFor="health-slider"
              className="block text-sm font-medium text-gray-400"
            >
              Simulate Motor Health Input
            </label>
            <input
              id="health-slider"
              type="range"
              min="0"
              max="100"
              value={motorHealth}
              onChange={(e) => setMotorHealth(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>Critical (0%)</span>
              <span>Optimal (100%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LearnMore;
