// Embedding Atlas — a 2D UMAP projection of the corpus (coords + clusters precomputed by the API).
// Pan/zoom/box-select the canvas, click a point to pull its embedding neighbours from the API, and
// inspect clusters in the side rail. Ported from the prototype with real data wired in.

"use client";

import { useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";

import { Icon } from "@/components/shared/Icon";
import { LatencyBadge } from "@/components/shared/LatencyBadge";
import { ErrorState, Loading } from "@/components/shared/States";
import { api } from "@/lib/api";
import { useTheme } from "@/lib/theme";
import { type AtlasPoint, type Category, type Neighbor } from "@/lib/types";
import { useApi } from "@/lib/useApi";

const VW = 1000;
const VH = 680;

interface Transform {
  k: number;
  x: number;
  y: number;
}

function AtlasView() {
  const theme = useTheme();
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialSel = searchParams.get("sel");

  const { data, error, loading, tookMs, refreshing } = useApi(
    useCallback((signal: AbortSignal) => api.getAtlas(signal), []),
    [],
  );

  const points = useMemo(() => data?.points ?? [], [data]);
  const categories = useMemo(() => data?.categories ?? [], [data]);
  const byId = useMemo(() => Object.fromEntries(points.map((point) => [point.id, point])), [points]);
  const colorOf = useMemo(() => {
    const map: Record<string, string> = {};
    categories.forEach((category) => (map[category.id] = category.color));
    return (clusterId: string) => map[clusterId] ?? "#888";
  }, [categories]);

  // Cluster centroids for the floating labels.
  const centroids = useMemo(() => {
    const acc: Record<string, { x: number; y: number; n: number }> = {};
    for (const point of points) {
      const entry = (acc[point.cluster_id] ??= { x: 0, y: 0, n: 0 });
      entry.x += point.x;
      entry.y += point.y;
      entry.n += 1;
    }
    Object.values(acc).forEach((entry) => {
      entry.x /= entry.n;
      entry.y /= entry.n;
    });
    return acc;
  }, [points]);

  const svgRef = useRef<SVGSVGElement>(null);
  const [tf, setTf] = useState<Transform>({ k: 1, x: 0, y: 0 });
  const [tool, setTool] = useState<"pan" | "select">("pan");
  // True while a pan gesture is active — drives the grab/grabbing cursor without reading the drag ref
  // during render.
  const [panning, setPanning] = useState(false);
  const [sel, setSel] = useState<string | null>(initialSel);
  const [hover, setHover] = useState<AtlasPoint | null>(null);
  // Tooltip position is computed from the live SVG rect when a point is hovered (in the pointer
  // handler), not during render — reading the DOM ref during render is both flagged and a layout sync.
  const [tipStyle, setTipStyle] = useState<React.CSSProperties>({ display: "none" });
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [box, setBox] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [region, setRegion] = useState<AtlasPoint[] | null>(null);
  const [query, setQuery] = useState("");
  const drag = useRef<{ mode: "pan"; sx: number; sy: number; tx: number; ty: number } | { mode: "box" } | null>(null);

  // Selected paper's embedding neighbours (fetched from the API on selection).
  const [neighbors, setNeighbors] = useState<Neighbor[]>([]);
  const selPoint = sel ? byId[sel] : null;

  const centerOn = useCallback((point: AtlasPoint, k = 2.2) => {
    setTf({ k, x: VW / 2 - point.x * k, y: VH / 2 - point.y * k });
  }, []);

  // Center the deep-linked paper once the atlas loads. `sel` already initializes to `initialSel`, so
  // the selection itself needs no set here; we only recenter the viewport (setTf, via centerOn) once
  // the point exists in `byId` — a load-time sync that has no render-derived equivalent.
  useEffect(() => {
    if (initialSel && byId[initialSel]) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      centerOn(byId[initialSel], 2.4);
    }
  }, [initialSel, byId, centerOn]);

  useEffect(() => {
    if (!sel) {
      // Clear stale neighbours when the selection is dropped; intentional reset, not a render-derived
      // value (the fetched list also feeds the link overlay).
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setNeighbors([]);
      return;
    }
    const controller = new AbortController();
    // The atlas is the SPECTER2 embedding projection, so its neighbour shortlist stays on the dense
    // recommender (matching the 2D layout) regardless of the global "More Like This" toggle.
    api
      .getNeighbors(sel, 7, "dense", controller.signal)
      .then(({ data: result }) => setNeighbors(result.neighbors))
      .catch(() => setNeighbors([]));
    return () => controller.abort();
  }, [sel]);

  const qLower = query.trim().toLowerCase();
  const matches = useCallback(
    (point: AtlasPoint) => qLower !== "" && point.title.toLowerCase().includes(qLower),
    [qLower],
  );
  const matchCount = useMemo(() => (qLower ? points.filter(matches).length : 0), [qLower, points, matches]);

  const selNeighborIds = useMemo(() => new Set(neighbors.map((n) => n.id)), [neighbors]);

  // ---- coordinate helpers ----
  const toData = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const vb = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    return { x: (vb.x - tf.x) / tf.k, y: (vb.y - tf.y) / tf.k };
  };

  const onWheel = (event: ReactWheelEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const pt = svg.createSVGPoint();
    pt.x = event.clientX;
    pt.y = event.clientY;
    const vb = pt.matrixTransform(svg.getScreenCTM()?.inverse());
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    setTf((t) => {
      const k = Math.min(9, Math.max(0.6, t.k * factor));
      const ratio = k / t.k;
      return { k, x: vb.x - (vb.x - t.x) * ratio, y: vb.y - (vb.y - t.y) * ratio };
    });
  };

  const onDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    const d = toData(event.clientX, event.clientY);
    if (tool === "select") {
      drag.current = { mode: "box" };
      setBox({ x0: d.x, y0: d.y, x1: d.x, y1: d.y });
      setRegion(null);
    } else {
      drag.current = { mode: "pan", sx: event.clientX, sy: event.clientY, tx: tf.x, ty: tf.y };
      setPanning(true);
    }
  };

  const onMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const dr = drag.current;
    if (!dr) return;
    if (dr.mode === "pan") {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const sx = (event.clientX - dr.sx) * (VW / rect.width);
      const sy = (event.clientY - dr.sy) * (VH / rect.height);
      setTf((t) => ({ ...t, x: dr.tx + sx, y: dr.ty + sy }));
    } else {
      const d = toData(event.clientX, event.clientY);
      setBox((b) => (b ? { ...b, x1: d.x, y1: d.y } : b));
    }
  };

  const onUp = () => {
    const dr = drag.current;
    drag.current = null;
    setPanning(false);
    if (dr?.mode === "box" && box) {
      const x0 = Math.min(box.x0, box.x1);
      const x1 = Math.max(box.x0, box.x1);
      const y0 = Math.min(box.y0, box.y1);
      const y1 = Math.max(box.y0, box.y1);
      if (Math.abs(x1 - x0) > 4 && Math.abs(y1 - y0) > 4) {
        const inside = points.filter(
          (p) => !hidden.has(p.cluster_id) && p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1,
        );
        setRegion(inside);
        setSel(null);
      } else {
        setBox(null);
      }
    }
  };

  const resetView = () => {
    setTf({ k: 1, x: 0, y: 0 });
    setBox(null);
    setRegion(null);
  };
  const zoom = (factor: number) =>
    setTf((t) => {
      const k = Math.min(9, Math.max(0.6, t.k * factor));
      const ratio = k / t.k;
      return { k, x: VW / 2 - (VW / 2 - t.x) * ratio, y: VH / 2 - (VH / 2 - t.y) * ratio };
    });
  const toggleCluster = (clusterId: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(clusterId)) next.delete(clusterId);
      else next.add(clusterId);
      return next;
    });

  const radiusOf = (p: AtlasPoint) => 4.6 + (p === selPoint ? 2.5 : 0);
  const opacityOf = (p: AtlasPoint) => {
    if (hidden.has(p.cluster_id)) return 0;
    if (qLower) return matches(p) ? 1 : 0.08;
    if (region) return region.includes(p) ? 1 : 0.16;
    if (selPoint) {
      if (p === selPoint) return 1;
      if (selNeighborIds.has(p.id)) return 1;
      return 0.22;
    }
    return 0.95;
  };

  const boxRect = box && {
    x: Math.min(box.x0, box.x1),
    y: Math.min(box.y0, box.y1),
    w: Math.abs(box.x1 - box.x0),
    h: Math.abs(box.y1 - box.y0),
  };

  const tipPos = (point: AtlasPoint) => {
    const svg = svgRef.current;
    if (!svg) return { display: "none" } as const;
    const rect = svg.getBoundingClientRect();
    const vx = point.x * tf.k + tf.x;
    const vy = point.y * tf.k + tf.y;
    const px = (vx / VW) * rect.width;
    const py = (vy / VH) * rect.height;
    return { left: Math.min(rect.width - 230, Math.max(8, px + 12)), top: Math.max(8, py - 10) };
  };

  if (error)
    return (
      <div className="feed-wrap">
        <ErrorState message={error} />
      </div>
    );
  if (loading || !data)
    return (
      <div className="feed-wrap">
        <Loading message="Projecting embeddings…" />
      </div>
    );

  return (
    <div className="atlas-wrap">
      <div className="atlas-canvas">
        <svg
          ref={svgRef}
          className="atlas-svg"
          viewBox={`0 0 ${VW} ${VH}`}
          preserveAspectRatio="xMidYMid meet"
          onWheel={onWheel}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={onUp}
          style={{ cursor: tool === "select" ? "crosshair" : panning ? "grabbing" : "grab" }}
        >
          <g transform={`translate(${tf.x} ${tf.y}) scale(${tf.k})`}>
            {/* neighbour links */}
            {selPoint &&
              neighbors.map((neighbor) => {
                const target = byId[neighbor.id];
                if (!target) return null;
                return (
                  <line
                    key={neighbor.id}
                    x1={selPoint.x}
                    y1={selPoint.y}
                    x2={target.x}
                    y2={target.y}
                    stroke="var(--accent)"
                    strokeWidth={1.1 / tf.k}
                    opacity={0.5}
                    strokeLinecap="round"
                  />
                );
              })}

            {/* points */}
            {points.map((point) => {
              const opacity = opacityOf(point);
              if (opacity === 0) return null;
              const isSel = point === selPoint;
              const isHov = point === hover;
              return (
                <circle
                  key={point.id}
                  cx={point.x}
                  cy={point.y}
                  r={radiusOf(point) / Math.sqrt(tf.k)}
                  fill={colorOf(point.cluster_id)}
                  opacity={opacity}
                  stroke={isSel ? "var(--atlas-sel)" : isHov ? "#fff" : "none"}
                  strokeWidth={(isSel ? 2.5 : 1.5) / tf.k}
                  style={{ cursor: "pointer", transition: "opacity .25s" }}
                  onPointerEnter={() => {
                    setHover(point);
                    setTipStyle(tipPos(point));
                  }}
                  onPointerLeave={() => setHover((h) => (h === point ? null : h))}
                  onClick={(event) => {
                    event.stopPropagation();
                    setSel(point.id);
                    setRegion(null);
                    setBox(null);
                  }}
                />
              );
            })}

            {/* cluster labels */}
            {!selPoint &&
              !region &&
              categories.map((category) => {
                if (hidden.has(category.id)) return null;
                const centroid = centroids[category.id];
                if (!centroid) return null;
                return (
                  <text
                    key={category.id}
                    x={centroid.x}
                    y={centroid.y}
                    className="atlas-clabel"
                    fill={category.color}
                    fontSize={15 / Math.sqrt(tf.k)}
                    opacity={0.9}
                    style={{
                      paintOrder: "stroke",
                      stroke: "var(--canvas-bg)",
                      strokeWidth: 4 / Math.sqrt(tf.k),
                    }}
                  >
                    {category.name}
                  </text>
                );
              })}

            {/* selection box */}
            {boxRect && (
              <rect
                x={boxRect.x}
                y={boxRect.y}
                width={boxRect.w}
                height={boxRect.h}
                fill="var(--accent)"
                fillOpacity={0.08}
                stroke="var(--accent)"
                strokeWidth={1.2 / tf.k}
                strokeDasharray={`${5 / tf.k} ${4 / tf.k}`}
              />
            )}
          </g>
        </svg>

        {/* hover tooltip */}
        {hover && (
          <div className="atlas-tip" style={tipStyle}>
            <div className="atlas-tip-t">{hover.title}</div>
            <div className="atlas-tip-m">arXiv:{hover.id}</div>
          </div>
        )}

        {/* tools */}
        <div className="atlas-tools">
          <div className="atlas-toolgrp">
            <button
              className={`atlas-tool${tool === "pan" ? " on" : ""}`}
              onClick={() => setTool("pan")}
              title="Pan & zoom"
            >
              <Icon name="compass" s={16} />
            </button>
            <button
              className={`atlas-tool${tool === "select" ? " on" : ""}`}
              onClick={() => setTool("select")}
              title="Box select"
            >
              <Icon name="lasso" s={16} />
            </button>
          </div>
          <div className="atlas-toolgrp">
            <button className="atlas-tool" onClick={() => zoom(1.25)} title="Zoom in">
              <Icon name="plus" s={16} />
            </button>
            <button className="atlas-tool" onClick={() => zoom(1 / 1.25)} title="Zoom out">
              <Icon name="minus" s={16} />
            </button>
            <button className="atlas-tool" onClick={resetView} title="Reset view">
              <Icon name="target" s={16} />
            </button>
          </div>
          <div className="atlas-zoomlvl">{tf.k.toFixed(1)}×</div>
        </div>

        {/* search overlay */}
        <div className="atlas-searchbar">
          <Icon name="search" s={16} c="var(--text-faint)" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Highlight points by title…"
          />
          {query && (
            <button className="atlas-clearq" onClick={() => setQuery("")}>
              <Icon name="x" s={14} />
            </button>
          )}
          {qLower && <span className="atlas-qcount">{matchCount} lit</span>}
        </div>

        <div className="atlas-lat">
          <LatencyBadge tookMs={tookMs} serverMs={null} busy={refreshing} show={theme.showLatency} />
        </div>
      </div>

      {/* inspector */}
      <aside className="atlas-inspector">
        <div className="atlas-insp-h">
          <h2>Embedding Atlas</h2>
          <p>
            UMAP projection of {data.count} papers. Drag to pan, scroll to zoom, click a point for its nearest
            neighbours.
          </p>
        </div>

        <div className="atlas-legend">
          <div className="atlas-legend-h">
            Clusters <span>click to toggle</span>
          </div>
          {categories.map((category: Category) => {
            const off = hidden.has(category.id);
            return (
              <button
                key={category.id}
                className={`atlas-leg${off ? " off" : ""}`}
                onClick={() => toggleCluster(category.id)}
              >
                <span className="atlas-leg-dot" style={{ background: category.color }} />
                <span className="atlas-leg-name">{category.name}</span>
                <span className="atlas-leg-n">{category.count}</span>
              </button>
            );
          })}
        </div>

        {selPoint ? (
          <div className="atlas-panel">
            <div className="atlas-panel-h">
              <span>Selected</span>
              <button className="atlas-x" onClick={() => setSel(null)}>
                <Icon name="x" s={14} />
              </button>
            </div>
            <h3 className="atlas-sel-t">{selPoint.title}</h3>
            <div className="atlas-sel-m">arXiv:{selPoint.id}</div>
            <div className="atlas-sel-acts">
              <button
                className="af-btn pri small"
                onClick={() => router.push(`/paper/${encodeURIComponent(selPoint.id)}`)}
              >
                Open paper
              </button>
              <button className="af-btn small" onClick={() => centerOn(selPoint, 2.6)}>
                Center
              </button>
            </div>
            <div className="atlas-nn-h">Nearest neighbors</div>
            {neighbors.map((neighbor) => (
              <button key={neighbor.id} className="atlas-nn" onClick={() => setSel(neighbor.id)}>
                <span className="atlas-nn-dot" style={{ background: colorOf(neighbor.cluster_id ?? "") }} />
                <span className="atlas-nn-sim">{neighbor.similarity.toFixed(2)}</span>
                <span className="atlas-nn-t">{neighbor.title}</span>
              </button>
            ))}
          </div>
        ) : region ? (
          <div className="atlas-panel">
            <div className="atlas-panel-h">
              <span>
                <b>{region.length}</b> papers in region
              </span>
              <button
                className="atlas-x"
                onClick={() => {
                  setRegion(null);
                  setBox(null);
                }}
              >
                <Icon name="x" s={14} />
              </button>
            </div>
            <div className="atlas-region-list">
              {region.slice(0, 40).map((point) => (
                <button key={point.id} className="atlas-region-row" onClick={() => setSel(point.id)}>
                  <span className="atlas-nn-dot" style={{ background: colorOf(point.cluster_id) }} />
                  <span className="atlas-region-t">{point.title}</span>
                </button>
              ))}
              {region.length > 40 && <div className="atlas-region-more">+{region.length - 40} more</div>}
            </div>
          </div>
        ) : (
          <div className="atlas-panel atlas-hint">
            <div className="atlas-hint-row">
              <Icon name="compass" s={16} c="var(--accent)" />
              Drag to pan · scroll to zoom
            </div>
            <div className="atlas-hint-row">
              <Icon name="dot" s={16} c="var(--accent)" />
              Click any point to see its nearest neighbours
            </div>
            <div className="atlas-hint-row">
              <Icon name="lasso" s={16} c="var(--accent)" />
              Switch to box-select to grab a region
            </div>
            <div className="atlas-hint-row">
              <Icon name="search" s={16} c="var(--accent)" />
              Search to highlight matching points
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

export default function AtlasPage() {
  return (
    <Suspense
      fallback={
        <div className="feed-wrap">
          <Loading message="Loading atlas…" />
        </div>
      }
    >
      <AtlasView />
    </Suspense>
  );
}
