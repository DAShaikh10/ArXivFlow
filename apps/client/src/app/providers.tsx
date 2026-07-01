// Client provider stack: theme prefs -> category lookup -> themed shell.

"use client";

import { type ReactNode } from "react";

import { AppShell } from "@/components/layout/AppShell";
import { CategoriesProvider } from "@/lib/categories";
import { ThemeProvider } from "@/lib/theme";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <CategoriesProvider>
        <AppShell>{children}</AppShell>
      </CategoriesProvider>
    </ThemeProvider>
  );
}
