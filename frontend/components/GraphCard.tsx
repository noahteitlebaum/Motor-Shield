"use client";

import { useState, ReactNode, useEffect } from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { motion } from "framer-motion";

export interface GraphCardProps {
  title: string;
  description: string;
  reactGraph?: ReactNode;
  type?: "vibration" | "current" | "temperature";
  isAnomaly?: boolean;
}

const SimulatedGraph = ({ type = "vibration", isAnomaly = false }) => {
  const [pathData, setPathData] = useState("");
  const [pathData2, setPathData2] = useState(""); // For multi-channel like current

  // Real sampled data from BLDC_OpenCircuit CSV (Ia and Ib, 60 samples over 80ms)
  // Ia: periodic bursts ~12.5A → normalised to SVG y (0=top, 100=bottom, 50=baseline)
  const realIa = [50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.7,50,50,50,50,9.7,50,50];
  const realIb = [50,50,9.6,50,50,50,50,9.7,50,50,50,50,9.7,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.8,50,50,50,50,9.9,50,50,50,50,9.9,50,50,50,50,9.9,50,50,50,50,9.9,50,50,50,50,9.9,50,50];
  // Vibration: derived from duty cycle switching pattern (da+db+dc -> mechanical load spikes)
  const realVib = [50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5,50,50,15,50,32.5];
  // Temperature: derived from total |Ia|+|Ib|+|Ic| magnitude (current → heat proxy)
  const realTemp = [70,70,42.9,70,57.1,70,70,42.9,70,57.1,70,70,42.9,70,57.1,70,70,42.9,70,57.1,70,70,43,70,57,70,70,43,70,57,70,70,43,70,57,70,70,43,70,57,70,70,43,70,57,70,70,43,70,57,70,70,43,70,57,70,70,43,70,57];

  useEffect(() => {
    let animationFrameId: number;
    let offset = 0;
    let realIdx = 0;

    const render = () => {
      // Speed up animation if it's an anomaly
      offset += isAnomaly ? 5.0 : 0.2;

      if (isAnomaly) {
        // Play back real CSV-derived data for all graph types
        realIdx = (realIdx + 1) % (realIa.length * 5);
        const sampleIdx = Math.floor(realIdx / 5);

        let points1 = "M 0 50";
        let points2 = "M 0 50";

        for (let i = 0; i < 60; i++) {
          const x = i * 5;
          const idx = (sampleIdx + i) % realIa.length;

          if (type === "current") {
            points1 += ` L ${x} ${realIa[idx]}`;
            points2 += ` L ${x} ${realIb[idx]}`;
          } else if (type === "vibration") {
            points1 += ` L ${x} ${realVib[idx]}`;
          } else if (type === "temperature") {
            points1 += ` L ${x} ${realTemp[idx]}`;
          }
        }

        setPathData(points1);
        if (type === "current") setPathData2(points2);
      } else {
        let points1 = "M 0 50";
        let points2 = "M 0 50";

        for (let i = 0; i <= 100; i += 2) {
          let y1 = 50;
          let y2 = 50;
          const x = i * 3;

          if (type === "vibration") {
            const burst = isAnomaly ? 35 : 5;

            y1 =
              50 +
              Math.sin(x * 0.5 + offset * 0.15) * burst +
              (Math.random() - 0.5) * (isAnomaly ? 10 : 2);
          } else if (type === "current") {
            const amp = isAnomaly ? 40 : 8;

            y1 =
              50 +
              Math.sin(x * 0.05 + offset * 0.1) * amp +
              (Math.random() - 0.5) * (isAnomaly ? 5 : 1.5);
            y2 =
              50 +
              Math.sin(x * 0.05 + offset * 0.1 + (Math.PI * 2) / 3) * amp +
              (Math.random() - 0.5) * (isAnomaly ? 5 : 1.5);
          } else if (type === "temperature") {
            const baseTemp = isAnomaly ? 80 : 40;

            y1 =
              100 -
              baseTemp +
              Math.sin(x * 0.02 + offset * 0.05) * (isAnomaly ? 5 : 2);
          }

          y1 = Math.max(5, Math.min(95, y1));
          y2 = Math.max(5, Math.min(95, y2));

          points1 += ` L ${x} ${y1}`;
          points2 += ` L ${x} ${y2}`;
        }

        setPathData(points1);
        if (type === "current") setPathData2(points2);
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationFrameId);
  }, [type, isAnomaly]);

  // Choose stroke color based on type
  const strokeColor =
    type === "vibration"
      ? "text-purple-500"
      : type === "current"
        ? "text-blue-500"
        : "text-danger-500";

  return (
    <div className="w-full h-full flex items-center justify-center p-4">
      <svg className="w-full h-full overflow-visible" viewBox="0 0 300 100">
        <line
          className="text-default-200"
          stroke="currentColor"
          strokeWidth="1"
          x1="0"
          x2="300"
          y1="20"
          y2="20"
        />
        <line
          className="text-default-200"
          stroke="currentColor"
          strokeWidth="1"
          x1="0"
          x2="300"
          y1="50"
          y2="50"
        />
        <line
          className="text-default-200"
          stroke="currentColor"
          strokeWidth="1"
          x1="0"
          x2="300"
          y1="80"
          y2="80"
        />

        <path
          className={strokeColor}
          d={pathData}
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
        />

        {type === "current" && (
          <path
            className="text-emerald-500"
            d={pathData2}
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
          />
        )}

        <path
          className={`${strokeColor}`}
          d={`${pathData} L 300 100 L 0 100 Z`}
          fill="currentColor"
          stroke="none"
          style={{ opacity: 0.15 }}
        />
      </svg>
    </div>
  );
};

export default function GraphCard({
  title,
  description,
  reactGraph,
  type = "vibration",
  isAnomaly = false,
}: GraphCardProps) {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div className="relative w-[300px] h-[300px] perspective-1000">
      <motion.div
        animate={{ rotateY: isFlipped ? 180 : 0 }}
        className="w-full h-full relative preserve-3d"
        initial={false}
        style={{ transformStyle: "preserve-3d" }}
        transition={{ duration: 0.6, animationDirection: "normal" }}
      >
        {/* Front Face */}
        <Card
          className={`absolute w-full h-full backface-hidden ${isFlipped ? "pointer-events-none" : ""} ${isAnomaly ? "border border-danger/50 shadow-danger/20 shadow-lg" : ""}`}
        >
          <CardHeader className="flex justify-between items-center z-10 w-full p-4">
            <div
              className={`bg-default-100 rounded-full px-4 py-1 shadow-sm mx-auto flex items-center gap-2 ${isAnomaly ? "border border-danger" : ""}`}
            >
              <div
                className={`w-2 h-2 rounded-full ${isAnomaly ? "bg-danger animate-ping" : "bg-success"} opacity-80`}
              />
              <p
                className={`font-bold text-default-600 truncate max-w-[150px] ${isAnomaly ? "text-danger" : ""}`}
              >
                {title}
              </p>
            </div>
            <Button
              isIconOnly
              className="absolute top-2 right-2 rounded-full hover:scale-110 transition-transform"
              size="sm"
              variant="flat"
              onPress={() => setIsFlipped(!isFlipped)}
            >
              ⟳
            </Button>
          </CardHeader>
          <CardBody className="flex justify-center items-center overflow-hidden p-0 relative">
            {reactGraph ? (
              reactGraph
            ) : (
              <SimulatedGraph isAnomaly={isAnomaly} type={type} />
            )}
          </CardBody>
        </Card>

        {/* Back Face */}
        <Card
          className={`absolute w-full h-full backface-hidden ${!isFlipped ? "pointer-events-none" : ""}`}
          style={{ transform: "rotateY(180deg)" }}
        >
          <CardHeader className="flex justify-between items-center z-10 w-full p-4">
            <div className="bg-default-100 rounded-full px-4 py-1 shadow-sm mx-auto">
              <p className="font-bold text-default-600 truncate max-w-[150px]">
                {title}
              </p>
            </div>
            <Button
              isIconOnly
              className="absolute top-2 right-2 rounded-full hover:scale-110 transition-transform"
              size="sm"
              variant="flat"
              onPress={() => setIsFlipped(!isFlipped)}
            >
              ⟳
            </Button>
          </CardHeader>
          <CardBody className="overflow-y-auto p-6 scrollbar-hide">
            <div className="bg-default-100 p-4 rounded-xl min-h-full">
              <p className="text-default-500 text-sm leading-relaxed">
                {description}
              </p>
            </div>
          </CardBody>
        </Card>
      </motion.div>
    </div>
  );
}
