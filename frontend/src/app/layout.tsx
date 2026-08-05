import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "MediShield — Case Management",
  description: "MediShield multi-agent document intake — case dashboard, review queue, and audit trail.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-50 font-sans">
        <header className="border-b border-slate-200 bg-white">
          <nav className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-4">
            <Link href="/" className="text-sm font-semibold tracking-tight text-slate-900">
              MediShield
            </Link>
            <Link href="/" className="text-sm text-slate-600 hover:text-slate-900">
              Dashboard
            </Link>
            <Link href="/review" className="text-sm text-slate-600 hover:text-slate-900">
              Review Queue
            </Link>
            <Link href="/upload" className="text-sm text-slate-600 hover:text-slate-900">
              Upload
            </Link>
          </nav>
        </header>
        <main className="flex-1">{children}</main>
      </body>
    </html>
  );
}
