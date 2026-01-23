"use client";

import { Avatar } from "@heroui/avatar";
import { Card, CardBody } from "@heroui/card";

type TeamMember = {
  name: string;
  role: string;
  bio: string;
  imgSrc: string;
};

const team: TeamMember[] = [
  {
    name: "Noah Teitlebaum",
    role: "Team Lead",
    bio: "role description",
    imgSrc: "",
  },
  {
    name: "Vedanshi Parekh",
    role: "Team Lead",
    bio: "role description",
    imgSrc: "",
  },
  {
    name: "Pratik Gupta",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Daniel He",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Adison Seo",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Krish Singh",
    role: "Backend/AI",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Aidan Naccarato",
    role: "Frontend",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Baichaun Ren",
    role: "Data/Eng",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Justin Jubinville",
    role: "Data/Eng",
    bio: "role description.",
    imgSrc: "",
  },
  {
    name: "Zayan Suri",
    role: "Data/Eng",
    bio: "role description.",
    imgSrc: "",
  },
];

export default function MeetTheTeam() {
  return (
    <div className="w-full flex flex-col items-center gap-8 py-8">
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Meet the Team</h1>
        <p className="text-default-500">The people building MotorShield.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-6xl px-4">
        {team.map((m) => (
          <Card
            key={m.name}
            className="p-4 hover:scale-[1.02] transition-transform shadow-sm"
          >
            <CardBody className="flex flex-row gap-4 items-center overflow-visible">
              <Avatar
                isBordered
                className="w-20 h-20 text-large"
                color="primary"
                name={m.name}
                size="lg"
                src={m.imgSrc}
              />
              <div className="flex flex-col gap-1">
                <h3 className="text-lg font-bold">{m.name}</h3>
                <p className="text-small text-blue-600 font-semibold uppercase">
                  {m.role}
                </p>
                <p className="text-small text-default-500">{m.bio}</p>
              </div>
            </CardBody>
          </Card>
        ))}
      </div>
    </div>
  );
}
