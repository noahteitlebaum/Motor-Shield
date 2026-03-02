"use client";

import { useState, ReactNode, useEffect } from "react";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Button } from "@heroui/button";
import { motion } from "framer-motion";

interface GraphCardProps {
  title: string;
  description: string;
  reactGraph: ReactNode;
}

const PlaceholderGraph = () => {
  // Generate a random path for a line graph look
  const [pathData, setPathData] = useState("");

  useEffect(() => {
    let points = "M 0 50";

    for (let i = 0; i <= 100; i += 5) {
      const y = 30 + Math.random() * 40; // Random height between 30 and 70

      points += ` L ${i * 3} ${y}`;
    }
    setPathData(points);
  }, []);

  return (
    <div className="w-full h-full flex items-center justify-center p-4">
      <svg className="w-full h-full overflow-visible" viewBox="0 0 300 100">
        {/* Grid lines */}
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

        {/* The Graph Line */}
        <path
          className="text-primary"
          d={pathData}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />

        {/* Gradient fill under line (optional, simple implementation) */}
        <path
          className="text-primary/20"
          d={`${pathData} L 300 100 L 0 100 Z`}
          fill="currentColor"
          stroke="none"
        />
      </svg>
    </div>
  );
};

export default function GraphCard({
  title,
  description,
  reactGraph,
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
        <Card className={`absolute w-full h-full backface-hidden ${isFlipped? "pointer-events-none" : ""}`}>
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
          <CardBody className="flex justify-center items-center overflow-hidden p-0">
            {reactGraph ? reactGraph : <PlaceholderGraph />}
          </CardBody>
        </Card>

        {/* Back Face */}
        <Card
          className={`absolute w-full h-full backface-hidden ${!isFlipped? "pointer-events-none" : ""}`}
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
