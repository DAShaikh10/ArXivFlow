// Latency badge — reports the *real* request timing.
//
// Total = client-measured round-trip (`tookMs`). When the endpoint also reports its server-side
// processing time (`serverMs`, from the API's `took_ms`), the popover splits total into server vs.
// network/parse so the number is honest rather than a simulated breakdown.

"use client";

import { useState } from "react";

import { fmtMs } from "@/lib/format";
import { Icon } from "./Icon";

interface LatencyBadgeProps {
  tookMs: number | null;
  serverMs?: number | null;
  busy?: boolean;
  show?: boolean;
  variant?: "pill" | "inline";
  label?: string;
}

function Popover({ tookMs, serverMs }: { tookMs: number; serverMs?: number | null }) {
  const stages: Array<[string, number]> =
    serverMs != null && serverMs <= tookMs
      ? [
          ["server", serverMs],
          ["network", Math.max(0, tookMs - serverMs)],
        ]
      : [["round-trip", tookMs]];

  return (
    <div className="af-lat-pop" onClick={(event) => event.stopPropagation()}>
      <div className="af-lat-pop-h">response breakdown</div>
      {stages.map(([key, value]) => (
        <div key={key} className="af-lat-stage">
          <span className="af-lat-k">{key}</span>
          <span className="af-lat-track">
            <i style={{ width: `${tookMs > 0 ? (value / tookMs) * 100 : 0}%` }} />
          </span>
          <span className="af-lat-v">{fmtMs(value)} ms</span>
        </div>
      ))}
      <div className="af-lat-tot">
        <span>total</span>
        <span>
          <b>{fmtMs(tookMs)}</b> ms
        </span>
      </div>
    </div>
  );
}

export function LatencyBadge({
  tookMs,
  serverMs,
  busy = false,
  show = true,
  variant = "pill",
  label = "ranked",
}: LatencyBadgeProps) {
  const [open, setOpen] = useState(false);
  if (!show) return null;

  const total = busy ? null : tookMs;

  if (variant === "inline") {
    return (
      <span className="af-lat-wrap">
        <button className="af-lat-inline" onClick={() => !busy && setOpen((value) => !value)}>
          {busy ? "computing…" : `${label} in ${fmtMs(total)} ms`}
          {!busy && (
            <Icon
              name="chevD"
              s={11}
              c="var(--text-faint)"
              style={{
                marginLeft: 4,
                verticalAlign: "middle",
                transform: open ? "rotate(180deg)" : "none",
                transition: ".2s",
              }}
            />
          )}
        </button>
        {open && !busy && total != null && <Popover tookMs={total} serverMs={serverMs ?? null} />}
      </span>
    );
  }

  return (
    <div className={`af-lat${busy ? " busy" : ""}`} onClick={() => !busy && setOpen((value) => !value)}>
      <span className={`af-lat-pulse${busy ? " spin" : ""}`} />
      <span className="af-lat-main">
        {busy ? (
          "computing…"
        ) : (
          <>
            <b>{fmtMs(total)}</b>
            {" "}ms
          </>
        )}
      </span>
      {!busy && (
        <Icon
          name="chevD"
          s={13}
          c="var(--text-faint)"
          style={{ transform: open ? "rotate(180deg)" : "none", transition: ".2s" }}
        />
      )}
      {open && !busy && total != null && <Popover tookMs={total} serverMs={serverMs ?? null} />}
    </div>
  );
}
