import type { Metadata } from "next";
import Image from "next/image";
import { Geist, Geist_Mono } from "next/font/google";
import { Montserrat } from 'next/font/google'
import "./globals.css";
import FooterBar from "./components/NavigationBar/FooterBar";

const montserrat = Montserrat({
  subsets: ['latin'],
  variable: '--font-montserrat',
  weight: ['100', '200', '300', '400', '500', '600', '700', '800', '900'],
})

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
      <body className={`${geistSans.variable} ${geistMono.variable} ${montserrat.variable} antialiased min-h-screen flex flex-col`}> {/*forcing body to be at least as tall as window; stacking elements vertically*/}

        {/* Top-left logo (site-wide) */}
        <header className="fixed top-2 left-4 z-60">
          <Link href="/" className="inline-block" aria-label="Go to home">
            <Image
              src="/MotorShieldLogo.png"
              alt="Motor Shield logo"
              className="w-20 h-20 object-contain"
              width={56}
              height={56}
              priority
            />
          </Link>
        </header>

        <main className={"flex-grow"}>
          {children}
        </main>
        
      <FooterBar/>
      </body>
    </html>
  );
}
