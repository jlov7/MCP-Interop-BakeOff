import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "usb-agents Console",
  description: "Local MCP portability lab for agent runtimes",
  icons: {
    icon: "/icon.svg"
  }
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
