import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Motorshield",
  description: "",
};

import Link from "next/link";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {/* Top-left logo (site-wide) */}
        <header className="fixed top-4 left-4 z-50">
          <Link href="/" className="inline-block" aria-label="Go to home">
            <img
              src="./MotorShieldLogo.png"
              alt="Motor Shield logo"
              className="w-20 h-20 object-contain"
              width={56}
              height={56}
            />
          </Link>
        </header>

        {children}
      </body>
    </html>
  );
}
