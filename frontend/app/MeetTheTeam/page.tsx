"use client";

import { Avatar } from "@heroui/avatar";
import { Card, CardBody } from "@heroui/card";
import FadeInUp from "../components/animations/FadeInUp";
import { Geist } from "next/font/google";
import { Linkedin } from 'lucide-react';
import { GithubIcon } from "@/components/icons";

type TeamMember = {
  name: string;
  role: string;
  LinkedinURL: string;
  GithubURL: string;
  imgSrc: string;
};

const team: TeamMember[] = [
  {
    name: "Noah Teitlebaum",
    role: "Team Lead",
    LinkedinURL: "https://www.linkedin.com/in/noahteitlebaum",
    GithubURL: "https://github.com/noahteitlebaum",
    imgSrc: "/NoahTeitlebaumHeadshot.jpeg",
  },
  {
    name: "Vedanshi Parekh",
    role: "Team Lead",
    LinkedinURL: "https://www.linkedin.com/in/vedanshiparekh/",
    GithubURL: "",
    imgSrc: "/VedanshiParekhHeadshot.jpg",
  },
  {
    name: "Pratik Gupta",
    role: "Backend/AI",
    LinkedinURL: "https://www.linkedin.com/in/pratikngupta",
    GithubURL: "https://github.com/pratikngupta",
    imgSrc: "/PratikGuptaHeadshot.jpeg",
  },
  {
    name: "Daniel He",
    role: "Backend/AI",
    LinkedinURL: "https://www.linkedin.com/in/daniel-he-1309b2294",
    GithubURL: "https://github.com/DanielHe09",
    imgSrc: "/DanielHeHeadshot.jpg",
  },
  {
    name: "Adison Seo",
    role: "Backend/AI",
    LinkedinURL: "https://www.linkedin.com/in/adison-seo-bb3a5a2b3",
    GithubURL: "",
    imgSrc: "/AdisonSeoHeadshot.jpg",
  },
  {
    name: "Krish Singh",
    role: "Backend/AI",
    LinkedinURL: "https://www.linkedin.com/in/krish-singh-a9a343299",
    GithubURL: "",
    imgSrc: "/KrishHeadshotCrop.jpg",
  },
  {
    name: "Aidan Naccarato",
    role: "Frontend",
    LinkedinURL: "https://www.linkedin.com/in/aidan-naccarato",
    GithubURL: "https://github.com/AidanNacc",
    imgSrc: "/AidanNaccaratoHeadshot.jpg",
  },
  {
    name: "Baichaun Ren",
    role: "Data/Eng",
    LinkedinURL: "https://www.linkedin.com/company/western-cyber-society",
    GithubURL: "",
    imgSrc: "/Baichuan RenHeadshot.jpeg",
  },
  {
    name: "Justin Jubinville",
    role: "Data/Eng",
    LinkedinURL: "https://www.linkedin.com/in/justinjubinville",
    GithubURL: "https://github.com/1jubinviljus",
    imgSrc: "/JustinJubinvilleHeadshot.jpg",
  },
  {
    name: "Zayan Suri",
    role: "Data/Eng",
    LinkedinURL: "https://www.linkedin.com/in/zayan-suri",
    GithubURL: "",
    imgSrc: "/ZayanSuriHeadshot.jpeg",
  },
];

export default function MeetTheTeam() {
  return (
    <div className="w-full flex flex-col items-center gap-8 py-8">
      <FadeInUp>
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Meet the Team</h1>
        <p className="text-default-500 pb-5">The people building MotorShield.</p>
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

            <div className="flex flex-row gap-3 mt-2">
               <a 
                    href={m.LinkedinURL} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="mt-1 text-[#0A66C2] hover:text-[#2583E0] transition-colors inline-flex"
                    aria-label={`${m.name}'s LinkedIn Profile`}
                  >
                    <Linkedin size={20} strokeWidth={2} />
                  </a>

                  <a 
                    href={m.GithubURL} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="mt-1 text-[#0A66C2] hover:text-[#2583E0] transition-colors inline-flex"
                    aria-label={`${m.name}'s Github Profile`}
                  >
                    <GithubIcon size={20} strokeWidth={2} />
                  </a>
            </div>
              </div>
            </CardBody>
          </Card>
        ))}
      </div>
      </FadeInUp>
    </div>
  );
}
