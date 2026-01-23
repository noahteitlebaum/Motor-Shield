"use client";

import { Progress } from "@heroui/progress";
import React from "react";

interface MotorHealthBarProps {
  health: number; // 0 to 100
}

export const MotorHealthBar: React.FC<MotorHealthBarProps> = ({ health }) => {
  // Clamp value between 0 and 100
  const clampedHealth = Math.min(100, Math.max(0, health));

  // Determine color based on health
  const getColor = (value: number) => {
    if (value > 66) return "success";
    if (value > 33) return "warning";

    return "danger";
  };

  return (
    <div className="w-full max-w-md mx-auto p-4 bg-default-50 dark:bg-default-100 rounded-2xl border border-default-200 shadow-md">
      <Progress
        classNames={{
          base: "max-w-md",
          track: "drop-shadow-md border border-default",
          indicator: "bg-gradient-to-r from-pink-500 to-yellow-500",
          label: "tracking-wider font-medium text-default-600",
          value: "text-foreground/60",
        }}
        color={getColor(clampedHealth)}
        label="Motor Health"
        showValueLabel={true}
        size="md"
        value={clampedHealth}
      />
    </div>
  );
};

export default MotorHealthBar;
