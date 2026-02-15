"use client";

import { Card, CardBody } from "@heroui/card";
import { Chip } from "@heroui/chip";
import FadeInUp from "../components/animations/FadeInUp";

export default function ProjectOverview() {
  return (
    <div className="w-full flex flex-col items-center gap-10 py-12 px-4">
      {/* Header */}
      <div className="max-w-4xl w-full flex flex-col gap-4">
        <FadeInUp delay={0.1}>
        <p className="text-sm font-bold text-blue-500 uppercase tracking-widest leading-none">
          MotorShield
        </p>
        </FadeInUp>
        <FadeInUp delay={0.2}>
        <h1 className="text-5xl font-black tracking-tighter text-foreground">
          Project Overview
        </h1>
        <p className="text-lg text-default-500 leading-relaxed max-w-2xl">
          MotorShield is an AI tool that predicts motor failure before it
          happens. Instead of waiting for breakdowns, we use motor signal data
          to estimate risk early, saving money and keeping systems running
          smoothly.
        </p>
        </FadeInUp>
      </div>

      {/* Why it matters */}
      <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-3 gap-6">
        <FadeInUp delay={0.3}>
        <Card className="p-4 bg-default-50">
          <CardBody>
            <h3 className="text-xl font-bold mb-2">Less downtime</h3>
            <p className="text-default-500 leading-relaxed">
              Unexpected motor failure can shut down an entire operation.
              MotorShield helps catch warning signs early so teams can act
              before things go down.
            </p>
          </CardBody>
        </Card>
        </FadeInUp>
        <FadeInUp delay={0.3}>
        <Card className="p-4 bg-default-50">
          <CardBody>
            <h3 className="text-xl font-bold mb-2">Lower cost</h3>
            <p className="text-default-500 leading-relaxed">
              Predicting issues early helps reduce emergency repair costs,
              delays, and part replacement caused by waiting too long.
            </p>
          </CardBody>
        </Card>
        </FadeInUp>
        <FadeInUp delay={0.3}>
        <Card className="p-4 bg-default-50">
          <CardBody>
            <h3 className="text-xl font-bold mb-2">Clear decisions</h3>
            <p className="text-default-500 leading-relaxed">
              Instead of raw sensor graphs, we summarize motor condition into a
              simple health score and trend that&apos;s easy to read fast.
            </p>
          </CardBody>
        </Card>
      </FadeInUp>
      </div>

      {/* How it works */}
      <div className="max-w-4xl w-full flex flex-col gap-8">
        <FadeInUp delay={0.4}>
        <div>
          <h2 className="text-3xl font-bold mb-4">How it works</h2>
          <p className="text-default-500 max-w-2xl">
            {" "}
            MotorShield follows a simple pipeline: simulate motor data, train
            the model, validate with sensors, and show results through a
            dashboard.
          </p>
        </div>
        </FadeInUp>

        {/* Visual Flow Diagram Placeholder */}
        <FadeInUp>
        <Card className="p-8 bg-default-50 items-center justify-center">
          <PipelineDiagram />
        </Card>
        </FadeInUp>

        <div className="flex flex-col gap-4">
          <FadeInUp>
          <Step
            desc="We generate motor behavior in a controlled environment. This creates realistic data with both healthy and failing conditions."
            icon={<SimulinkIcon />}
            num="1"
            title="Simulink Simulation"
          />
          </FadeInUp>
          <FadeInUp>
          <Step
            desc="We train a model to detect patterns that appear before failure happens. Our current best approach is a CNN -> LSTM pipeline for time-series data."
            icon={<AIModelIcon />}
            num="2"
            title="Deep Learning Model"
          />
          </FadeInUp>
          <FadeInUp>
          <Step
            desc="Eventually, we'll collect real motor readings using sensors and a microcontroller, then run predictions in real time to validate performance outside simulation."
            icon={<SensorIcon />}
            num="3"
            title="Microcontroller + Sensors"
          />
          </FadeInUp>
          <FadeInUp>
          <Step
            desc="The web app visualizes motor health, trends, and failure risk in a clean format that's easy to understand."
            icon={<DashboardIcon />}
            num="4"
            title="Website Dashboard"
          />
          </FadeInUp>
        </div>
      </div>

      {/* Example output visual */}
      <Card className="max-w-4xl w-full p-8 bg-default-50">
        <FadeInUp>
        <div className="flex justify-between items-start mb-6">
          <div>
            <p className="text-xs font-bold text-blue-500 uppercase mb-1">
              Example Output
            </p>
            <h3 className="text-2xl font-bold">Motor Health</h3>
          </div>
          <Chip
            className="font-bold border-blue-500 text-foreground"
            color="primary"
            variant="bordered"
          >
            Demo
          </Chip>
        </div>

        <div className="flex justify-between items-end mb-4">
          <div>
            <p className="text-6xl font-black text-blue-500 leading-none mb-1">
              82
            </p>
            <p className="text-sm text-default-500">out of 100</p>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-blue-500 uppercase mb-1">
              Status
            </p>
            <p className="text-lg font-bold text-success">Healthy</p>
          </div>
        </div>

        <div className="h-3 w-full bg-default-200 rounded-full overflow-hidden">
          <div className="h-full bg-success w-[82%] rounded-full" />
        </div>
      </FadeInUp>

      </Card>

    </div>
  );
}

function Step({
  num,
  title,
  desc,
  icon,
}: {
  num: string;
  title: string;
  desc: string;
  icon?: React.ReactNode;
}) {
  return (
    <Card className="p-6 bg-default-50">
      <div className="flex gap-6 items-start">
        <div className="flex flex-col items-center gap-3 shrink-0">
          <div className="flex items-center justify-center w-12 h-12 rounded-xl border-2 border-blue-500 bg-background text-lg font-black text-foreground">
            {num}
          </div>
          {icon && (
            <div className="w-20 h-20 flex items-center justify-center bg-background rounded-xl border border-default-200 p-3">
              {icon}
            </div>
          )}
        </div>
        <div>
          <h3 className="text-xl font-bold mb-2">{title}</h3>
          <p className="text-default-500 leading-relaxed">{desc}</p>
        </div>
      </div>
    </Card>
  );
}

// Icons
function SimulinkIcon() {
  return (
    <svg
      fill="none"
      height="60"
      viewBox="0 0 60 60"
      width="60"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        fill="#6299BB"
        height="20"
        opacity="0.3"
        rx="4"
        width="30"
        x="15"
        y="20"
      />
      <circle cx="30" cy="30" fill="#6299BB" r="8" />
      <circle
        className="text-background"
        cx="30"
        cy="30"
        fill="currentColor"
        r="4"
      />
      <path
        d="M10 15 Q15 20 20 15 T30 15 T40 15"
        fill="none"
        stroke="#6299BB"
        strokeWidth="2"
      />
      <path
        d="M10 25 Q15 30 20 25 T30 25 T40 25"
        fill="none"
        stroke="#6299BB"
        strokeWidth="2"
      />
      <path
        d="M20 35 Q25 40 30 35 T40 35"
        fill="none"
        stroke="#6299BB"
        strokeWidth="2"
      />
      <path
        d="M45 30 L52 30 M50 28 L52 30 L50 32"
        fill="none"
        stroke="#4a9d63"
        strokeWidth="2"
      />
    </svg>
  );
}

function AIModelIcon() {
  return (
    <svg
      fill="none"
      height="60"
      viewBox="0 0 60 60"
      width="60"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g fill="#6299BB">
        <circle cx="15" cy="15" r="4" />
        <circle cx="30" cy="10" r="4" />
        <circle cx="45" cy="15" r="4" />
        <circle cx="15" cy="30" r="4" />
        <circle cx="30" cy="25" r="4" />
        <circle cx="45" cy="30" r="4" />
        <circle cx="20" cy="45" r="4" />
        <circle cx="30" cy="40" r="4" />
        <circle cx="40" cy="45" r="4" />
      </g>
      <circle cx="30" cy="52" fill="#4a9d63" r="4" />

      <g opacity="0.4" stroke="#6299BB" strokeWidth="1.5">
        <line x1="15" x2="30" y1="15" y2="25" />
        <line x1="30" x2="30" y1="10" y2="25" />
        <line x1="45" x2="30" y1="15" y2="25" />
        <line x1="30" x2="20" y1="25" y2="45" />
        <line x1="30" x2="30" y1="25" y2="40" />
        <line x1="30" x2="40" y1="25" y2="45" />
      </g>
      <line stroke="#4a9d63" strokeWidth="2" x1="30" x2="30" y1="40" y2="52" />
    </svg>
  );
}

function SensorIcon() {
  return (
    <svg
      fill="none"
      height="60"
      viewBox="0 0 60 60"
      width="60"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        fill="#6299BB"
        height="18"
        opacity="0.3"
        rx="2"
        width="24"
        x="18"
        y="25"
      />
      <rect fill="#6299BB" height="14" rx="1" width="20" x="20" y="27" />
      <rect
        className="text-background"
        fill="currentColor"
        height="8"
        width="12"
        x="24"
        y="30"
      />
      <line
        stroke="#6299BB"
        strokeWidth="0.5"
        x1="27"
        x2="33"
        y1="32"
        y2="32"
      />
      <line
        stroke="#6299BB"
        strokeWidth="0.5"
        x1="27"
        x2="33"
        y1="34"
        y2="34"
      />
      <line
        stroke="#6299BB"
        strokeWidth="0.5"
        x1="27"
        x2="33"
        y1="36"
        y2="36"
      />
      <circle cx="12" cy="20" fill="#4a9d63" r="3" />
      <circle cx="48" cy="20" fill="#4a9d63" r="3" />
      <circle cx="12" cy="48" fill="#4a9d63" r="3" />
      <circle cx="48" cy="48" fill="#4a9d63" r="3" />
      <g opacity="0.6" stroke="#4a9d63" strokeWidth="1.5">
        <line x1="15" x2="20" y1="20" y2="28" />
        <line x1="45" x2="38" y1="20" y2="28" />
        <line x1="15" x2="20" y1="48" y2="40" />
        <line x1="45" x2="38" y1="48" y2="40" />
      </g>
    </svg>
  );
}

function DashboardIcon() {
  return (
    <svg
      fill="none"
      height="60"
      viewBox="0 0 60 60"
      width="60"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect
        fill="#6299BB"
        height="36"
        opacity="0.2"
        rx="2"
        width="44"
        x="8"
        y="12"
      />
      <rect
        className="text-background"
        fill="currentColor"
        height="32"
        rx="1"
        width="40"
        x="10"
        y="14"
      />
      <g fill="#6299BB">
        <rect height="6" width="4" x="14" y="38" />
        <rect height="12" width="4" x="20" y="32" />
        <rect height="16" width="4" x="26" y="28" />
      </g>
      <g fill="#4a9d63">
        <rect height="10" width="4" x="32" y="34" />
        <rect height="14" width="4" x="38" y="30" />
        <rect height="8" width="4" x="44" y="36" />
      </g>
      <text
        fill="#6299BB"
        fontSize="8"
        fontWeight="bold"
        textAnchor="middle"
        x="30"
        y="25"
      >
        82
      </text>
      <circle cx="20" cy="24" fill="#4a9d63" r="2" />
    </svg>
  );
}

function PipelineDiagram() {
  return (
    <div className="flex items-center justify-between flex-wrap gap-4 w-full max-w-2xl">
      <div className="flex flex-col items-center flex-1 min-w-[100px]">
        <div className="w-16 h-16 rounded-xl bg-background border-2 border-blue-500 flex items-center justify-center mb-2">
          <SimulinkIcon />
        </div>
        <p className="text-xs font-bold text-blue-500 uppercase tracking-wider">
          Simulate
        </p>
      </div>

      <div className="w-8 h-[2px] bg-default-300" />

      <div className="flex flex-col items-center flex-1 min-w-[100px]">
        <div className="w-16 h-16 rounded-xl bg-background border-2 border-blue-500 flex items-center justify-center mb-2">
          <AIModelIcon />
        </div>
        <p className="text-xs font-bold text-blue-500 uppercase tracking-wider">
          Learn
        </p>
      </div>

      <div className="w-8 h-[2px] bg-default-300" />

      <div className="flex flex-col items-center flex-1 min-w-[100px]">
        <div className="w-16 h-16 rounded-xl bg-background border-2 border-blue-500 flex items-center justify-center mb-2">
          <SensorIcon />
        </div>
        <p className="text-xs font-bold text-blue-500 uppercase tracking-wider">
          Collect
        </p>
      </div>

      <div className="w-8 h-[2px] bg-default-300" />

      <div className="flex flex-col items-center flex-1 min-w-[100px]">
        <div className="w-16 h-16 rounded-xl bg-background border-2 border-blue-500 flex items-center justify-center mb-2">
          <DashboardIcon />
        </div>
        <p className="text-xs font-bold text-blue-500 uppercase tracking-wider">
          Predict
        </p>
      </div>
    </div>
  );
}
