// Circular similarity gauge (0..1), ported from the prototype.

interface SimRingProps {
  value: number;
  size?: number;
  stroke?: number;
  label?: boolean;
}

export function SimRing({ value, size = 54, stroke = 5, label = true }: SimRingProps) {
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, value));

  return (
    <div className="af-ring" style={{ width: size, height: size, position: "relative" }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="var(--ring-track)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset .5s cubic-bezier(.2,.7,.2,1)" }}
        />
      </svg>
      {label && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            fontFamily: "var(--mono)",
            fontWeight: 600,
            fontSize: size * 0.24,
            letterSpacing: "-0.02em",
          }}
        >
          {Math.round(clamped * 100)}
        </div>
      )}
    </div>
  );
}
