"use client";

import { Card, CardBody } from "@heroui/card";

export default function MotorStatus() {
  return (
    <Card className="h-full min-h-[250px] bg-black">
      <CardBody className="p-0 flex items-center justify-center">
        <div className="flex flex-col items-center gap-2 text-default-500">
          {/* Camera/Video Off Icon */}
          <svg
            fill="none"
            height="48"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="1"
            viewBox="0 0 24 24"
            width="48"
          >
            <path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2m5.66 0H14a2 2 0 0 1 2 2v3.34l1 1L23 7v10" />
            <line x1="1" x2="23" y1="1" y2="23" />
          </svg>
          <p className="text-sm font-mono uppercase tracking-widest">
            No Signal
          </p>
        </div>
      </CardBody>
    </Card>
  );
}
