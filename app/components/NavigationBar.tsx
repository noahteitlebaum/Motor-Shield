"use client";

import NavButton from "./NavButton";
import styles from "./NavButton.module.css";
import { usePathname } from "next/navigation";

const routes = [
  { name: "Dashboard", link: "/Dashboard" },
  { name: "Learn More", link: "/LearnMore" },
  { name: "Meet the Team", link: "/MeetTheTeam" },
  { name: "Project Overview", link: "/ProjectOverview" },
];

export default function NavigationBar() {
  const pathname = usePathname();

  return (
    <>
      <nav className={styles.navContainer}>
        {routes.map((r) => (
          <NavButton key={r.link} name={r.name} link={r.link} active={pathname === r.link} />
        ))}
      </nav>
      <div className={styles.navSpacer} aria-hidden />
    </>
  );
}