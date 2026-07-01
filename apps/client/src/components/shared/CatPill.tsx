// A coloured cluster pill. Resolves the cluster name + colour from the categories context; renders
// nothing when the paper is unclustered (e.g. before the atlas projection has been built).

"use client";

import { useCategories } from "@/lib/categories";

interface CatPillProps {
  clusterId: string | null | undefined;
  soft?: boolean;
  size?: "sm" | "md";
}

export function CatPill({ clusterId, soft = true, size = "sm" }: CatPillProps) {
  const { byId } = useCategories();
  if (!clusterId) return null;

  const category = byId[clusterId];
  const color = category?.color ?? "#888";
  const fontSize = size === "sm" ? 11 : 12;

  return (
    <span
      className="af-cat"
      style={{
        color,
        background: soft ? `${color}1a` : "transparent",
        border: soft ? "none" : `1px solid ${color}66`,
        fontSize,
      }}
    >
      {category?.name ?? clusterId}
    </span>
  );
}
