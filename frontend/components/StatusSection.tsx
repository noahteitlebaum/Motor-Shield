"use client";

import { useState, useEffect } from "react";
import { Card, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";

import AlertsPanel from "./AlertsPanel";
import MotorStatus from "./MotorStatus";

interface StatusSectionProps {
  diagnosis?: string;
  latestLog?: { type: string, message: string } | null;
}

export default function StatusSection({ diagnosis = "Healthy", latestLog }: StatusSectionProps) {
  const [inferenceTime, setInferenceTime] = useState(12);

  // Slightly fluctuate inference time
  useEffect(() => {
    const interval = setInterval(() => {
      setInferenceTime(12 + Math.floor(Math.random() * 5) - 2);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  const isHealthy = diagnosis === "Healthy";

  return (
    <Card className={`w-full min-h-[400px] mb-5 p-6 transition-colors duration-500 bg-default-50 border-none`}>
      <CardBody className="flex flex-col lg:flex-row justify-between gap-6 h-full">
        {/* Left: General Status Text */}
        <div className="flex-1 flex flex-col gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <p className="text-2xl font-bold">AI Diagnostics</p>
              <Chip color="success" variant="flat" size="sm" className="mb-1">
                LIVE
              </Chip>
            </div>
            <div className="h-1 w-20 rounded-full bg-success transition-colors" />
          </div>

          <div className="flex flex-col gap-3 mt-2 text-sm text-default-600 border border-default-200 p-4 rounded-xl bg-background/50">
            <div className="flex justify-between items-center border-b border-default-100 pb-2">
              <span className="font-semibold">Active Model</span>
              <span className="text-purple-500 font-mono font-bold">Hybrid CNN-Transformer</span>
            </div>
            <div className="flex justify-between items-center border-b border-default-100 pb-2">
              <span className="font-semibold">Window Size</span>
              <span className="font-mono">200 samples (40ms)</span>
            </div>
            <div className="flex justify-between items-center border-b border-default-100 pb-2">
              <span className="font-semibold">Feature Input</span>
              <span className="font-mono">6 Channels (3I, 3V)</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="font-semibold">Inference Speed</span>
              <span className="font-mono text-blue-500">{inferenceTime}ms</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-auto">
            <div className="p-4 bg-background rounded-xl border border-default-200 flex flex-col items-center">
              <p className="text-xs text-default-500 uppercase font-bold text-center">
                System Health
              </p>
              <p className={`text-2xl font-mono font-bold text-center mt-1 ${isHealthy ? "text-success" : "text-danger"}`}>
                {isHealthy ? "OPTIMAL" : "CRITICAL"}
              </p>
            </div>
            <div className="p-4 bg-background rounded-xl border border-default-200 flex flex-col items-center">
              <p className="text-xs text-default-500 uppercase font-bold text-center">
                Data Points Processed
              </p>
              <p className="text-2xl font-mono font-bold text-primary text-center mt-1">
                14.2M
              </p>
            </div>
          </div>
        </div>

        {/* Center: Alerts */}
        <div className="flex-1">
          <AlertsPanel latestLog={latestLog} />
        </div>

        {/* Right: Motor Status Visual */}
        <div className="flex-1">
          <MotorStatus diagnosis={diagnosis} />
        </div>
      </CardBody>
    </Card>
  );
}
