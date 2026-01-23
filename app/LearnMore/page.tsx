"use client";

import React, { useState } from "react";
import { Slider } from "@heroui/slider";
import { Card, CardBody, CardHeader } from "@heroui/card";

import MotorHealthBar from "../../components/MotorHealthBar";

export default function LearnMore() {
  const [motorHealth, setMotorHealth] = useState<number>(85);

  return (
    <div className="w-full flex flex-col items-center gap-8 py-8">
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-emerald-500 bg-clip-text text-transparent">
          Learn More
        </h1>
        <p className="text-default-500">
          Understanding system diagnostics and health monitoring.
        </p>
      </div>

      <Card className="w-full max-w-2xl p-4">
        <CardHeader className="flex flex-col gap-2 items-center">
          <h2 className="text-2xl font-semibold">System Diagnostics</h2>
          <p className="text-default-500 text-center">
            Real-time monitoring of motor performance and structural integrity.
          </p>
        </CardHeader>
        <CardBody className="gap-8">
          {/* Motor Health Visualization */}
          <div className="py-4 flex justify-center w-full">
            <MotorHealthBar health={motorHealth} />
          </div>

          {/* Simulation Controls */}
          <div className="space-y-6 pt-6 border-t border-default-200">
            <div className="flex flex-col gap-2">
              <p className="text-sm font-medium text-default-600">
                Simulate Motor Health Input
              </p>
              <Slider
                className="max-w-md mx-auto"
                color="success"
                defaultValue={85}
                label="Health Level"
                maxValue={100}
                minValue={0}
                showSteps={true}
                size="sm"
                step={1}
                value={motorHealth}
                onChange={(v) => setMotorHealth(v as number)}
              />
              <div className="flex justify-between text-xs text-default-400 max-w-md mx-auto w-full px-1">
                <span>Critical (0%)</span>
                <span>Optimal (100%)</span>
              </div>
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
