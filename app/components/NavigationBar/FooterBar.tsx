"use client";

import { AlignHorizontalJustifyStart } from "lucide-react";
import NavButton from "./NavButton";
import styles from "./NavButton.module.css";
import { usePathname } from "next/navigation";

const routes = [
  { name: "Meet the Team", link: "/MeetTheTeam" },
  { name: "Project Overview", link: "/ProjectOverview" },
];

export default function FooterBar() {
  const pathname = usePathname();

  return (
    <>
      <nav className={styles.footerContainer}>
        <div className={styles.footerLogo}>
          <img
            src="./MotorShieldLogo.png"
            alt="Motor Shield logo"
            className="w-15 h-15 object-contain"
            width={56}
            height={56}
          />

          <p
            style={{
              color: "var(--text-secondary)",
              alignSelf: "center",
              textAlign: "center",
              margin: "0px 10px",
            }}
          >
            Motorshield
            <br />
            2026
          </p>
        </div>

        <a href="mailto:anacca@uwo.ca" className={styles.emailLink}>
          Contact
          <img src="./mail.svg" className="w-8 h-6.5 object-contain" />
        </a>

        {routes.map((r) => (
          <NavButton
            key={r.link}
            name={r.name}
            link={r.link}
            active={pathname === r.link}
          />
        ))}
      </nav>
    </>
  );
}
