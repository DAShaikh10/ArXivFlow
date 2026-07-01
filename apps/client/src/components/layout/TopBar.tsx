// Top navigation bar: brand, primary nav (Discover / Atlas / Search), the search teaser,
// a quick dark-mode toggle and the preferences popover.

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { Icon, type IconName } from "@/components/shared/Icon";
import { useTheme } from "@/lib/theme";

interface NavLink {
  href: string;
  icon: IconName;
  label: string;
  badge?: string;
  matchPrefixes: string[];
}

const NAV: NavLink[] = [
  { href: "/", icon: "compass", label: "Discover", matchPrefixes: ["/", "/paper"] },
  { href: "/atlas", icon: "scatter", label: "Embedding Atlas", matchPrefixes: ["/atlas"] },
  {
    href: "/search",
    icon: "search",
    label: "Search",
    matchPrefixes: ["/search"],
  },
];

function isActive(pathname: string, link: NavLink): boolean {
  if (link.href === "/") return pathname === "/" || pathname.startsWith("/paper");
  return link.matchPrefixes.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

export function TopBar() {
  const pathname = usePathname();
  const theme = useTheme();
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <header className="af-topbar">
      <Link className="af-brand" href="/">
        <span className="af-mark">
          <Icon name="scatter" s={18} c="#fff" />
        </span>
        <span className="af-brand-t">
          arXiv<b>Flow</b>
        </span>
      </Link>

      <nav className="af-nav">
        {NAV.map((link) => (
          <Link key={link.href} href={link.href} className={`nav-item${isActive(pathname, link) ? " on" : ""}`}>
            <Icon name={link.icon} s={17} />
            {link.label}
            {link.badge && <span className="nav-badge">{link.badge}</span>}
          </Link>
        ))}
      </nav>

      <Link className="af-headsearch" href="/search" title="Search the corpus">
        <Icon name="search" s={15} c="var(--text-faint)" />
        <span>Ask in plain language…</span>
      </Link>

      <button className="af-theme" onClick={() => theme.set("dark", !theme.dark)} title="Toggle theme">
        <Icon name={theme.dark ? "sun" : "moon"} s={17} />
      </button>

      <div style={{ position: "relative" }}>
        <button
          className="af-theme"
          onClick={() => setSettingsOpen((open) => !open)}
          title="Preferences"
          aria-haspopup="dialog"
          aria-expanded={settingsOpen}
        >
          <Icon name="sliders" s={17} />
        </button>
        {settingsOpen && <SettingsPanel onClose={() => setSettingsOpen(false)} />}
      </div>
    </header>
  );
}
