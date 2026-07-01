// The themed application shell: the `.af-root` flex column (top bar + stage) that wraps every page.
// The atlas view is full-bleed (no scroll); every other view scrolls.

"use client";

import { usePathname } from "next/navigation";
import { type ReactNode } from "react";

import { useTheme } from "@/lib/theme";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: ReactNode }) {
  const theme = useTheme();
  const pathname = usePathname();
  const isAtlas = pathname.startsWith("/atlas");

  return (
    <div className={theme.rootClassName} data-theme={theme.dark ? "dark" : "light"} style={theme.rootStyle}>
      <TopBar />
      <main className={`af-stage${isAtlas ? " atlas-mode" : ""}`}>
        <div className={isAtlas ? "stage-full" : "stage-scroll"}>{children}</div>
      </main>
    </div>
  );
}
