"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import { Card, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";

import FadeInUp from "../components/animations/FadeInUp";

import GraphCard from "@/components/GraphCard";
import StatusSection from "@/components/StatusSection";

export type AnomalyType = "vibration" | "current" | "temperature" | null;

const ALL_CLASSES = [
  "Healthy",
  "Faulty_Control_Switch",
  "Faulty_Inter_Turn",
  "Faulty_Open_Circuit",
] as const;

export default function Dashboard() {
  const [diagnosis, setDiagnosis] = useState<string>("Faulty_Open_Circuit");
  const [activeAnomaly, setActiveAnomaly] = useState<AnomalyType>(null);
  const [latestLog, setLatestLog] = useState<{
    type: string;
    message: string;
  } | null>(null);

  // Use a ref so the interval always reads latest diagnosis without restarting
  const diagnosisRef = React.useRef(diagnosis);
  useEffect(() => { diagnosisRef.current = diagnosis; }, [diagnosis]);

  useEffect(() => {
    // Orchestrator loop
    const logInterval = setInterval(() => {
      const current = diagnosisRef.current;

      // Trigger anomaly event based on active diagnosis
      if (current === "Healthy") {
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
      } else if (current === "Faulty_Open_Circuit") {
        setActiveAnomaly("current");
        setLatestLog({
          type: "danger",
          message: "Possible open circuit — check wiring. Motor status: CRITICAL.",
        });
      } else if (current === "Faulty_Inter_Turn") {
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
      } else if (current === "Faulty_Control_Switch") {
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

      // Only clear anomaly for non-critical faults (open circuit stays permanently spiked)
      if (current !== "Faulty_Open_Circuit") {
        setTimeout(() => {
          setActiveAnomaly(null);
        }, 2000);
      }
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
            description="Frequency domain analysis of motor vibrations showing potential bearing faults detected at 120Hz harmonic."
            isAnomaly={activeAnomaly === "vibration"}
            title="Vibration Analysis"
            type="vibration"
          />
          <GraphCard
            description="Real-time phase current monitoring. Spikes indicate increased load or potential short circuits in the windings."
            isAnomaly={activeAnomaly === "current"}
            title="Current Draw"
            type="current"
          />
          <GraphCard
            description="Stator winding temperature readings. Sustained high temperatures may lead to insulation degradation."
            isAnomaly={activeAnomaly === "temperature"}
            title="Temperature"
            type="temperature"
          />
        </div>
      </FadeInUp>

              <Card className="w-full mb-6 border-none bg-default-50/50 shadow-sm backdrop-blur-md">
          <CardBody className="flex flex-col md:flex-row items-center gap-8 p-6">
            <div className="shrink-0 bg-white rounded-2xl p-4 flex items-center justify-center shadow-inner w-full md:w-auto h-full max-w-[250px] aspect-square">
              <Image src="/TransMotorShieldMotor.png" alt="Motor used in the project" width={200} height={200} className="object-contain" />
            </div>
            <div className="flex flex-col gap-3">
              <h2 className="text-2xl font-bold tracking-tight">System Configuration</h2>
              <p className="text-default-500 leading-relaxed text-sm">
                The MotorShield system is operating within nominal parameters, utilizing real-time telemetry to maintain winding temperatures and torque consistency for optimal efficiency. Integration with the hardware backend ensures every micro-step is logged, while predictive maintenance algorithms scan for harmonic distortions to preemptively mitigate mechanical fatigue. By synchronizing with the primary controller, the system manages voltage regulation and thermal dissipation, effectively neutralizing recent excursions in Sector 4 to prevent unplanned downtime and extend the lifespan of all connected motor units.
              </p>
            </div>
          </CardBody>
        </Card>
    </div>
  );
}
