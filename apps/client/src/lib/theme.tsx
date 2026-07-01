// Runtime theming context.
//
// Theme switching is done with CSS custom properties (see styles/themes/_tokens.scss): the provider
// keeps the user's preferences, persists them to localStorage, and injects --accent / font-family
// variables onto the `.af-root` wrapper. Everything else derives from those variables in SCSS.

"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { type Recommender } from "./types";

export const ACCENTS = ["#FFDDAE", "#6D94C5", "#9a3b2e", "#0d7d6b", "#3f3f46"] as const;

export const RECOMMENDERS: Array<{ key: Recommender; label: string; hint: string }> = [
  { key: "dense", label: "Semantic", hint: "SPECTER2 embedding similarity" },
  { key: "bm25", label: "Keyword", hint: "BM25 lexical similarity over title + abstract" },
];

export type FontPair = "editorial" | "technical" | "grotesk";
export type Density = "compact" | "comfortable";

export const FONTS: Record<FontPair, { sans: string; serif: string; body: string; mono: string }> = {
  editorial: {
    sans: '"Spline Sans"',
    serif: '"Source Serif 4"',
    body: '"Newsreader"',
    mono: '"IBM Plex Mono"',
  },
  technical: {
    sans: '"IBM Plex Sans"',
    serif: '"IBM Plex Sans"',
    body: '"IBM Plex Sans"',
    mono: '"IBM Plex Mono"',
  },
  grotesk: {
    sans: '"Spline Sans"',
    serif: '"Space Grotesk"',
    body: '"Spline Sans"',
    mono: '"Space Grotesk"',
  },
};

export interface ThemePrefs {
  dark: boolean;
  accent: string;
  density: Density;
  fontPair: FontPair;
  showLatency: boolean;
  recCount: number;
  recommender: Recommender;
}

const DEFAULTS: ThemePrefs = {
  dark: false,
  accent: ACCENTS[0],
  density: "comfortable",
  fontPair: "editorial",
  showLatency: true,
  recCount: 6,
  recommender: "dense",
};

const STORAGE_KEY = "arxivflow.prefs";

interface ThemeContextValue extends ThemePrefs {
  set: <K extends keyof ThemePrefs>(key: K, value: ThemePrefs[K]) => void;
  rootClassName: string;
  rootStyle: React.CSSProperties;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [prefs, setPrefs] = useState<ThemePrefs>(DEFAULTS);

  // Hydrate from localStorage after mount (avoids SSR mismatch). This deliberately sets state in an
  // effect: a lazy useState initializer would read localStorage during the server-less first render
  // and reintroduce the hydration mismatch we're avoiding. The rule-satisfying fix is a
  // useSyncExternalStore rewrite of the provider — out of scope for a dependency bump.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      if (stored) setPrefs({ ...DEFAULTS, ...(JSON.parse(stored) as Partial<ThemePrefs>) });
    } catch {
      // ignore malformed storage
    }
  }, []);

  const set = useMemo(
    () =>
      <K extends keyof ThemePrefs>(key: K, value: ThemePrefs[K]) =>
        setPrefs((prev) => {
          const next = { ...prev, [key]: value };
          try {
            window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
          } catch {
            // ignore quota / disabled storage
          }
          return next;
        }),
    [],
  );

  const value = useMemo<ThemeContextValue>(() => {
    const fonts = FONTS[prefs.fontPair];
    return {
      ...prefs,
      set,
      rootClassName: `af-root density-${prefs.density}${prefs.dark ? " dark" : ""}`,
      rootStyle: {
        ["--accent" as string]: prefs.accent,
        ["--accent-soft" as string]: `${prefs.accent}1a`,
        ["--sans" as string]: `${fonts.sans}, system-ui, sans-serif`,
        ["--serif" as string]: `${fonts.serif}, Georgia, serif`,
        ["--body-serif" as string]: `${fonts.body}, Georgia, serif`,
        ["--mono" as string]: `${fonts.mono}, ui-monospace, monospace`,
      },
    };
  }, [prefs, set]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within a ThemeProvider");
  return ctx;
}
