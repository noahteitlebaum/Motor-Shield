"use client";

import Link from "next/link";
import styles from "./NavButton.module.css";

type Props = {
  name: string;
  link: string;
  active?: boolean;
};

export default function NavButton({ name, link, active = false }: Props) {
  return (
    <Link
      href={link}
      className={`${styles.button} ${active ? styles.active : ""}`}
      aria-current={active ? "page" : undefined}
    >
      <span className={styles.label}>{name}</span>
    </Link>
  );
}