// Inline SVG icon set (ported from the prototype's icons.js, typed). Stroke icons inherit
// currentColor unless `c` is given.

import { type CSSProperties, type ReactNode } from "react";

export type IconName =
  | "search"
  | "ext"
  | "list"
  | "chevD"
  | "chevR"
  | "chevL"
  | "arrowL"
  | "compass"
  | "scatter"
  | "bookmark"
  | "clock"
  | "sun"
  | "moon"
  | "lock"
  | "expand"
  | "collapse"
  | "sort"
  | "filter"
  | "target"
  | "plus"
  | "minus"
  | "lasso"
  | "x"
  | "dot"
  | "sliders"
  | "tag";

interface IconProps {
  name: IconName;
  s?: number;
  sw?: number;
  c?: string;
  fill?: string;
  style?: CSSProperties;
  className?: string;
}

const PATHS: Record<IconName, (c?: string) => ReactNode> = {
  search: () => (
    <>
      <circle cx={11} cy={11} r={7} />
      <path d="m20 20-3.5-3.5" />
    </>
  ),
  ext: () => (
    <>
      <path d="M14 4h6v6" />
      <path d="M20 4 10 14" />
      <path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />
    </>
  ),
  list: () => <path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01" />,
  chevD: () => <path d="m6 9 6 6 6-6" />,
  chevR: () => <path d="m9 6 6 6-6 6" />,
  chevL: () => <path d="m15 6-6 6 6 6" />,
  arrowL: () => (
    <>
      <path d="M19 12H5" />
      <path d="m12 19-7-7 7-7" />
    </>
  ),
  compass: () => (
    <>
      <circle cx={12} cy={12} r={9} />
      <path d="m15.5 8.5-2 5-5 2 2-5z" />
    </>
  ),
  scatter: () => (
    <>
      <circle cx={6} cy={7} r={1.6} />
      <circle cx={12} cy={5} r={1.6} />
      <circle cx={17} cy={9} r={1.6} />
      <circle cx={8} cy={14} r={1.6} />
      <circle cx={15} cy={16} r={1.6} />
      <circle cx={19} cy={18} r={1.6} />
    </>
  ),
  bookmark: () => <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />,
  clock: () => (
    <>
      <circle cx={12} cy={12} r={9} />
      <path d="M12 7v5l3 2" />
    </>
  ),
  sun: () => (
    <>
      <circle cx={12} cy={12} r={4} />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: () => <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  lock: () => (
    <>
      <rect x={4} y={11} width={16} height={9} rx={2} />
      <path d="M8 11V7a4 4 0 0 1 8 0v4" />
    </>
  ),
  expand: () => (
    <path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3" />
  ),
  collapse: () => (
    <path d="M3 8V5a2 2 0 0 1 2-2h3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M21 16v3a2 2 0 0 0-2 2h-3" />
  ),
  sort: () => <path d="M3 6h11M3 12h7M3 18h4M18 8V20M18 20l3-3M18 20l-3-3" />,
  filter: () => <path d="M3 5h18l-7 8v6l-4-2v-4z" />,
  target: () => (
    <>
      <circle cx={12} cy={12} r={8} />
      <circle cx={12} cy={12} r={3.5} />
    </>
  ),
  plus: () => <path d="M12 5v14M5 12h14" />,
  minus: () => <path d="M5 12h14" />,
  lasso: () => (
    <path d="M4 12c0-4 3.6-7 8-7s8 3 8 7-3.6 7-8 7c-1 0-2-.2-2.8-.5M6 19a2 2 0 1 0 0 .01M6 17c0-2 1-3 3-3" />
  ),
  x: () => <path d="M18 6 6 18M6 6l12 12" />,
  dot: (c) => <circle cx={12} cy={12} r={4} fill={c ?? "currentColor"} />,
  sliders: () => (
    <>
      <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3" />
      <path d="M1 14h6M9 8h6M17 16h6" />
    </>
  ),
  tag: () => (
    <>
      <path d="M20.59 13.41 13.42 20.6a2 2 0 0 1-2.83 0L3 13V3h10l7.59 7.59a2 2 0 0 1 0 2.82z" />
      <path d="M7 7h.01" />
    </>
  ),
};

export function Icon({ name, s = 18, sw = 1.7, c = "currentColor", fill = "none", style, className }: IconProps) {
  const render = PATHS[name];
  if (!render) return null;
  return (
    <svg
      width={s}
      height={s}
      viewBox="0 0 24 24"
      fill={fill}
      stroke={c}
      strokeWidth={sw}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...(style ? { style } : {})}
      {...(className ? { className } : {})}
      aria-hidden
    >
      {render(c)}
    </svg>
  );
}
