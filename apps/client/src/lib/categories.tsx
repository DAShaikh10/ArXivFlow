// Shared cluster/category lookup.
//
// Papers carry no native arXiv category, so the backend groups them into embedding clusters (see
// apps/api/scripts/build_atlas.py). We fetch that legend once at app start and expose a id->Category
// map so CatPill and other views can colour/name a paper from its `cluster_id`.

"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { api } from "./api";
import { type Category } from "./types";

interface CategoriesContextValue {
  list: Category[];
  byId: Record<string, Category>;
}

const CategoriesContext = createContext<CategoriesContextValue>({ list: [], byId: {} });

export function CategoriesProvider({ children }: { children: ReactNode }) {
  const [list, setList] = useState<Category[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    api
      .getCategories(controller.signal)
      .then(({ data }) => setList(data))
      .catch(() => {
        // Categories are optional (need the atlas build); the UI degrades gracefully without them.
      });
    return () => controller.abort();
  }, []);

  const value = useMemo<CategoriesContextValue>(
    () => ({ list, byId: Object.fromEntries(list.map((category) => [category.id, category])) }),
    [list],
  );

  return <CategoriesContext.Provider value={value}>{children}</CategoriesContext.Provider>;
}

export function useCategories(): CategoriesContextValue {
  return useContext(CategoriesContext);
}
