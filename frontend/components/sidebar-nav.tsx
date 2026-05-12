"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';

import { NAV_ITEMS } from '../lib/types';

export function SidebarNav() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-title">Stock Analyzer</div>
        <p className="sidebar-copy">ML-based stock scoring and analysis</p>
      </div>

      <nav className="nav-stack" aria-label="Primary navigation">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link key={item.href} href={item.href} className={`nav-link${active ? ' active' : ''}`}>
              <span className="nav-link-label">{item.label}</span>
              <span className="nav-link-copy">Open</span>
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer card">
        <div className="sidebar-footer-row">
          <span className="badge">API</span>
          <span className="status ok">Ready</span>
        </div>
      </div>
    </aside>
  );
}
