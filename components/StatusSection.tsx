"use client";

import { Card, CardBody } from "@heroui/card";

import AlertsPanel from "./AlertsPanel";
import MotorStatus from "./MotorStatus";

export default function StatusSection() {
  return (
    <Card className="w-full min-h-[400px] mb-5 p-6 bg-default-50">
      <CardBody className="flex flex-col lg:flex-row justify-between gap-6 h-full">
        {/* Left: General Status Text */}
        <div className="flex-1 flex flex-col gap-4">
          <div>
            <p className="text-2xl font-bold mb-2">System Status</p>
            <div className="h-1 w-20 bg-blue-500 rounded-full" />
          </div>

          <div className="prose prose-sm dark:prose-invert text-default-500 max-w-none">
            <p>
            The MotorShield system is operating within nominal parameters, 
            utilizing real-time telemetry to maintain winding temperatures and torque 
            consistency for optimal efficiency. Integration with the hardware backend ensures 
            every micro-step is logged, while predictive maintenance algorithms scan for harmonic 
            distortions to preemptively mitigate mechanical fatigue. By synchronizing with the primary 
            controller, the system manages voltage regulation and thermal dissipation—effectively neutralizing 
            recent excursions in Sector 4 to prevent unplanned downtime and extend the lifespan of all connected motor units.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-auto">
            <div className="p-4 bg-background rounded-xl border border-default-200">
              <p className="text-xs text-default-500 uppercase font-bold">
                Uptime
              </p>
              <p className="text-2xl font-mono font-bold text-success">99.9%</p>
            </div>
            <div className="p-4 bg-background rounded-xl border border-default-200">
              <p className="text-xs text-default-500 uppercase font-bold">
                Efficiency
              </p>
              <p className="text-2xl font-mono font-bold text-primary">92.4%</p>
            </div>
          </div>
        </div>

        {/* Center: Alerts */}
        <div className="flex-1">
          <AlertsPanel />
        </div>

        {/* Right: Motor Status Visual */}
        <div className="flex-1">
          <MotorStatus />
        </div>
      </CardBody>
    </Card>
  );
}
