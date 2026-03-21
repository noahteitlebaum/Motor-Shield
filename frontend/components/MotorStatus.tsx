"use client";

import { useState, useEffect } from "react";
import { Card, CardBody } from "@heroui/card";
import { CircularProgress } from "@heroui/progress";

export default function MotorStatus({
  diagnosis = "Healthy",
}: {
  diagnosis?: string;
}) {
  const [health, setHealth] = useState(85);

  useEffect(() => {
    let baseVal = 85;

    if (diagnosis === "Healthy") baseVal = 97;
    else if (diagnosis === "Faulty_Control_Switch") baseVal = 68;
    else if (diagnosis === "Faulty_Inter_Turn") baseVal = 42;
    else if (diagnosis === "Faulty_Open_Circuit") baseVal = 0;

    // Slightly fluctuate health during simulation
    const interval = setInterval(() => {
      setHealth(baseVal + (Math.random() * 4 - 2));
    }, 1200);

    return () => clearInterval(interval);
  }, [diagnosis]);

  return (
    <Card className="h-full min-h-[250px] transition-colors duration-500 border-none bg-background">
      <CardBody className="p-6 flex flex-col items-center justify-center">
        <h3 className="font-bold text-center mb-6 text-lg tracking-wide uppercase">
          AI Diagnostic Score
        </h3>

        <div className="relative flex items-center justify-center">
          <CircularProgress
            classNames={{
              svg: "w-36 h-36 drop-shadow-md relative z-10",
              indicator: `transition-all duration-300 ${health > 80 ? "stroke-success" : health > 50 ? "stroke-warning" : "stroke-danger"}`,
              track: "stroke-default-200",
              value: "text-3xl font-bold text-foreground",
            }}
            color={health > 80 ? "success" : health > 50 ? "warning" : "danger"}
            showValueLabel={true}
            strokeWidth={4}
            value={health}
          />
        </div>

        <div className="mt-8 flex flex-col items-center gap-1">
          <p className="text-sm text-default-600 font-mono font-bold bg-default-100 px-3 py-1 rounded-md">
            Diagnosis: {diagnosis.replace(/_/g, " ")}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <div className={`w-2 h-2 rounded-full bg-primary animate-ping`} />
            <p className={`text-xs font-bold uppercase text-primary`}>
              Receiving Telemetry...
            </p>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}
