import type { Metadata } from 'next';
import { SidebarNav } from '../components/sidebar-nav';
import './globals.css';

export const metadata: Metadata = {
  title: 'Alpha Intelligence | Market Desk',
  description: 'Production-style dark dashboard for metrics, inference, and screener workflows powered by FastAPI.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="app-frame">
          <SidebarNav />
          <div className="app-content">{children}</div>
        </div>
      </body>
    </html>
  );
}
