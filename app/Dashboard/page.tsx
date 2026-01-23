"use client";

import GraphCard from "@/components/GraphCard";
import StatusSection from "@/components/StatusSection";

export default function Dashboard() {
  return (
    <div className="w-full flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-default-500">
          Monitor real-time motor status and analytics.
        </p>
      </div>

      <StatusSection />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 justify-items-center">
        <GraphCard
          description="Frequency domain analysis of motor vibrations showing potential bearing faults detected at 120Hz harmonic."
          reactGraph={null}
          title="Vibration Analysis"
        />
        <GraphCard
          description="Real-time phase current monitoring. Spikes indicate increased load or potential short circuits in the windings."
          reactGraph={null}
          title="Current Draw"
        />
        <GraphCard
          description="Stator winding temperature readings. Sustained high temperatures may lead to insulation degradation."
          reactGraph={null}
          title="Temperature"
        />
      </div>
    </div>
  );
}
