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
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
              eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
              enim ad minim veniam, quis nostrud exercitation ullamco laboris
              nisi ut aliquip ex ea commodo consequat.
            </p>
            <p>
              Duis aute irure dolor in reprehenderit in voluptate velit esse
              cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat
              cupidatat non proident, sunt in culpa qui officia deserunt mollit
              anim id est laborum.
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
