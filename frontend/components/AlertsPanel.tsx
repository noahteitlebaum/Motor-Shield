"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardBody } from "@heroui/card";
import { Divider } from "@heroui/divider";
import { motion, AnimatePresence } from "framer-motion";

export default function AlertsPanel({
  latestLog,
}: {
  latestLog?: { type: string; message: string } | null;
}) {
  const [alerts, setAlerts] = useState([
    {
      id: 1,
      type: "success",
      message: "System calibration completed successfully.",
      time: "2m ago",
    },
    {
      id: 2,
      type: "default",
      message: "Routine maintenance scheduled for tomorrow.",
      time: "1h ago",
    },
  ]);

  useEffect(() => {
    if (!latestLog) return;
    setAlerts((prev) =>
      [{ id: Date.now(), ...latestLog, time: "Just now" }, ...prev].slice(0, 5),
    );
  }, [latestLog]);

  return (
    <Card className="h-full min-h-[250px] max-h-[400px] border-none shadow-none bg-transparent">
      <CardHeader className="flex gap-3 justify-center pb-2 pt-5">
        <div className="flex flex-col items-center">
          <p className="text-md font-bold text-center text-xl uppercase tracking-wider flex items-center gap-2">
            Event Log
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-danger opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-danger" />
            </span>
          </p>
          <div className="h-1 w-12 bg-warning rounded-full mt-1" />
        </div>
      </CardHeader>
      <Divider />
      <CardBody className="overflow-y-auto px-2 scrollbar-hide">
        <div className="flex flex-col gap-2">
          <AnimatePresence initial={false}>
            {alerts.length === 0 ? (
              <p className="text-center text-default-500 text-sm mt-4">
                No recent events.
              </p>
            ) : (
              alerts.map((alert) => (
                <motion.div
                  key={alert.id}
                  layout
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  className="flex gap-3 items-start p-3 rounded-lg hover:bg-default-100 transition-colors border border-transparent hover:border-default-200"
                  exit={{ opacity: 0, scale: 0.9 }}
                  initial={{ opacity: 0, y: -20, scale: 0.95 }}
                  transition={{ duration: 0.3 }}
                >
                  <div
                    className={`w-2 h-2 mt-2 rounded-full shrink-0 ${
                      alert.type === "danger"
                        ? "bg-danger"
                        : alert.type === "warning"
                          ? "bg-warning"
                          : alert.type === "success"
                            ? "bg-success"
                            : alert.type === "primary"
                              ? "bg-primary"
                              : "bg-default-400"
                    }`}
                  />
                  <div className="flex flex-col gap-1 w-full">
                    <p className="text-small text-foreground leading-tight pr-2">
                      {alert.message}
                    </p>
                    <p className="text-tiny text-default-400 font-mono tracking-tighter">
                      {alert.time}
                    </p>
                  </div>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </CardBody>
    </Card>
  );
}
