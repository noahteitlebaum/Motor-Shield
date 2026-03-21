"use client";

import { useState, useEffect } from "react";
import GraphCard from "@/components/GraphCard";
import StatusSection from "@/components/StatusSection";
import FadeInUp from "../components/animations/FadeInUp";

export type AnomalyType = "vibration" | "current" | "temperature" | null;

export default function Dashboard() {
  const [diagnosis, setDiagnosis] = useState("Healthy");
  const [activeAnomaly, setActiveAnomaly] = useState<AnomalyType>(null);
  const [latestLog, setLatestLog] = useState<{
    type: string;
    message: string;
  } | null>(null);

  useEffect(() => {
    // Pick diagnosis on mount
    const classes = [
      "Healthy",
      "Faulty_Control_Switch",
      "Faulty_Inter_Turn",
      "Faulty_Open_Circuit",
    ];
    const pickedStatus = classes[Math.floor(Math.random() * classes.length)];
    setDiagnosis(pickedStatus);

    // Orchestrator loop
    const logInterval = setInterval(() => {
      // 60% chance to just output a normal system log rather than an anomaly
      if (Math.random() > 0.4) {
        setActiveAnomaly(null);
        const normalLogs = [
          "Inference block completed. Health stable.",
          "Telemetry received.",
          "Routine diagnostic check passed.",
          "Acoustic signature within normal bounds.",
          "Network latency 12ms.",
        ];
        setLatestLog({
          type: "success",
          message: normalLogs[Math.floor(Math.random() * normalLogs.length)],
        });
        return;
      }

      // 40% chance to trigger an anomaly based on diagnosis
      if (pickedStatus === "Healthy") {
        setActiveAnomaly(null);
        const healthyLogs = [
          "Motor efficiency optimal.",
          "Power factor stable at 0.95.",
          "Cooling system performing well.",
        ];
        setLatestLog({
          type: "success",
          message: healthyLogs[Math.floor(Math.random() * healthyLogs.length)],
        });
      } else if (pickedStatus === "Faulty_Open_Circuit") {
        setActiveAnomaly("current");
        const msgs = [
          "Phase C current dropped to zero unexpectedly.",
          "Severe current imbalance detected.",
          "Missing phase cycle detected by AI.",
        ];
        setLatestLog({
          type: "danger",
          message: msgs[Math.floor(Math.random() * msgs.length)],
        });
      } else if (pickedStatus === "Faulty_Inter_Turn") {
        if (Math.random() > 0.5) {
          setActiveAnomaly("temperature");
          const msgs = [
            "Rapid thermal increase detected in stator.",
            "Hotspot localized in Winding B.",
            "Insulation integrity warning triggered.",
          ];
          setLatestLog({
            type: "warning",
            message: msgs[Math.floor(Math.random() * msgs.length)],
          });
        } else {
          setActiveAnomaly("current");
          const msgs = [
            "Current imbalance detected across phases.",
            "Excessive current draw during operation gap.",
          ];
          setLatestLog({
            type: "danger",
            message: msgs[Math.floor(Math.random() * msgs.length)],
          });
        }
      } else if (pickedStatus === "Faulty_Control_Switch") {
        setActiveAnomaly("vibration");
        const msgs = [
          "Abnormal relay switching translating to rotor vibration.",
          "Transient spike caught in vibration data.",
          "Unexpected high-frequency harmonics.",
        ];
        setLatestLog({
          type: "warning",
          message: msgs[Math.floor(Math.random() * msgs.length)],
        });
      }

      // Clear anomaly shortly after to create a "spike" effect rather than continuous
      setTimeout(() => {
        setActiveAnomaly(null);
      }, 2000);
    }, 3500);

    return () => clearInterval(logInterval);
  }, []);

  return (
    <div className="w-full flex flex-col gap-6">
      <FadeInUp>
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
          <div className="flex flex-col gap-2">
            <h1 className="text-4xl font-bold tracking-tight">Dashboard</h1>
            <p className="text-default-500 pb-5">
              Simulated motor telemetry and diagnostics.
            </p>
          </div>
        </div>

        <StatusSection diagnosis={diagnosis} latestLog={latestLog} />
      </FadeInUp>

      <FadeInUp>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 justify-items-center pb-10 mt-4">
          <GraphCard
            title="Vibration Analysis"
            description="Frequency domain analysis of motor vibrations showing potential bearing faults detected at 120Hz harmonic."
            type="vibration"
            isAnomaly={activeAnomaly === "vibration"}
          />
          <GraphCard
            title="Current Draw"
            description="Real-time phase current monitoring. Spikes indicate increased load or potential short circuits in the windings."
            type="current"
            isAnomaly={activeAnomaly === "current"}
          />
          <GraphCard
            title="Temperature"
            description="Stator winding temperature readings. Sustained high temperatures may lead to insulation degradation."
            type="temperature"
            isAnomaly={activeAnomaly === "temperature"}
          />
        </div>
      </FadeInUp>
    </div>
  );
}
