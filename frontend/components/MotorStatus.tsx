"use client";

import { Card, CardBody } from "@heroui/card";
import Image from "next/image";

export default function MotorStatus() {
  return (
    <Card className="h-full min-h-[250px] bg-white">
      <CardBody className="p-0 flex items-center justify-center">
        <div className="flex flex-col items-center gap-2 text-default-500">
          {/*Image of the motor we are using (from Amazon) */}
          <Image
          src="/TransMotorShieldMotor.png"
          alt="Picture of the motor used for the project."
          width="300"
          height="300"
          />
        </div>
      </CardBody>
    </Card>
  );
}
