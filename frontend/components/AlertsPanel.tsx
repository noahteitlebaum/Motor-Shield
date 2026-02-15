"use client";

import { Card, CardHeader, CardBody } from "@heroui/card";
import { Divider } from "@heroui/divider";

export default function AlertsPanel() {
  const alerts = [
    {
      id: 1,
      type: "warning",
      message: "Vibration threshold exceeded in Sector 4.",
      time: "10m ago",
    },
    {
      id: 2,
      type: "success",
      message: "System calibration completed successfully.",
      time: "1h ago",
    },
    {
      id: 3,
      type: "danger",
      message: "Temperature spike detected on Motor B.",
      time: "2h ago",
    },
    {
      id: 4,
      type: "default",
      message: "Routine maintenance scheduled for tomorrow.",
      time: "5h ago",
    },
  ];

  return (
    <Card className="h-full max-h-[400px]">
      <CardHeader className="flex gap-3 justify-center pb-2">
        <div className="flex flex-col items-center">
          <p className="text-md font-bold text-center text-xl uppercase tracking-wider">
            Recent Alerts
          </p>
          <div className="h-1 w-12 bg-warning rounded-full mt-1" />
        </div>
      </CardHeader>
      <Divider />
      <CardBody className="overflow-y-auto px-2">
        <div className="flex flex-col gap-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="flex gap-3 items-start p-3 rounded-lg hover:bg-default-100 transition-colors"
            >
              <div
                className={`w-2 h-2 mt-2 rounded-full shrink-0 ${
                  alert.type === "danger"
                    ? "bg-danger"
                    : alert.type === "warning"
                      ? "bg-warning"
                      : alert.type === "success"
                        ? "bg-success"
                        : "bg-default-400"
                }`}
              />
              <div className="flex flex-col gap-1">
                <p className="text-small text-foreground leading-tight">
                  {alert.message}
                </p>
                <p className="text-tiny text-default-400">{alert.time}</p>
              </div>
            </div>
          ))}
          <div className="p-3">
            <p className="text-small text-default-500 italic text-center">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Investiga
              tiones demonstraverunt lectores legere me lius quod ii legunt
              saepius.
            </p>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}
