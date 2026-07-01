// Preferences popover (theme, accent, density, fonts, latency, recommendation count). Promoted from
// the prototype's design-tool "Tweaks" shell into a real, persisted settings control.

"use client";

import { useEffect, useRef } from "react";

import { ACCENTS, type Density, type FontPair, RECOMMENDERS, useTheme } from "@/lib/theme";

const FONT_OPTIONS: FontPair[] = ["editorial", "technical", "grotesk"];
const DENSITY_OPTIONS: Density[] = ["compact", "comfortable"];

export function SettingsPanel({ onClose }: { onClose: () => void }) {
  const theme = useTheme();
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click / Escape.
  useEffect(() => {
    function onPointer(event: MouseEvent) {
      if (ref.current && !ref.current.contains(event.target as Node)) onClose();
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div className="af-settings-pop" ref={ref}>
      <div className="af-settings-sec">Theme</div>
      <div className="af-settings-row">
        <label>Dark mode</label>
        <button
          className={`af-switch${theme.dark ? " on" : ""}`}
          role="switch"
          aria-checked={theme.dark}
          aria-label="Dark mode"
          onClick={() => theme.set("dark", !theme.dark)}
        />
      </div>
      <div className="af-settings-row">
        <label>Accent</label>
        <div className="af-swatches">
          {ACCENTS.map((accent) => (
            <button
              key={accent}
              className={`af-swatch${theme.accent === accent ? " on" : ""}`}
              style={{ background: accent }}
              aria-label={`Accent ${accent}`}
              onClick={() => theme.set("accent", accent)}
            />
          ))}
        </div>
      </div>

      <div className="af-settings-sec">Layout</div>
      <div className="af-settings-row">
        <label>Density</label>
        <div className="af-seg">
          {DENSITY_OPTIONS.map((density) => (
            <button
              key={density}
              className={theme.density === density ? "on" : ""}
              onClick={() => theme.set("density", density)}
            >
              {density}
            </button>
          ))}
        </div>
      </div>
      <div className="af-settings-row">
        <label>Latency badges</label>
        <button
          className={`af-switch${theme.showLatency ? " on" : ""}`}
          role="switch"
          aria-checked={theme.showLatency}
          aria-label="Latency badges"
          onClick={() => theme.set("showLatency", !theme.showLatency)}
        />
      </div>

      <div className="af-settings-sec">Recommendations</div>
      <div className="af-settings-row">
        <label>Recommender</label>
        <div className="af-seg">
          {RECOMMENDERS.map((rec) => (
            <button
              key={rec.key}
              className={theme.recommender === rec.key ? "on" : ""}
              title={rec.hint}
              onClick={() => theme.set("recommender", rec.key)}
            >
              {rec.label}
            </button>
          ))}
        </div>
      </div>
      <div className="af-settings-row">
        <label>Similar papers shown</label>
        <input
          type="range"
          min={3}
          max={12}
          step={1}
          value={theme.recCount}
          onChange={(event) => theme.set("recCount", Number(event.target.value))}
        />
        <span className="af-settings-val">{theme.recCount}</span>
      </div>

      <div className="af-settings-sec">Typography</div>
      <div className="af-settings-row">
        <label>Font pairing</label>
        <select value={theme.fontPair} onChange={(event) => theme.set("fontPair", event.target.value as FontPair)}>
          {FONT_OPTIONS.map((font) => (
            <option key={font} value={font}>
              {font}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
