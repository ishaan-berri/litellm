/**
 * useItemNavigation — generic keyboard navigation for detail drawers.
 *
 * Handles J/K (next/prev) and Escape (close). Works with any item type
 * via a caller-supplied `getId` function.
 *
 * Used by: LogDetailsDrawer (via request_id), WorkflowRunDrawer (via run_id)
 */

import { useCallback, useEffect } from "react";

interface UseItemNavigationProps<T> {
  isOpen: boolean;
  currentItem: T | null;
  allItems: T[];
  getId: (item: T) => string;
  onClose: () => void;
  onSelect?: (item: T) => void;
}

export function useItemNavigation<T>({
  isOpen,
  currentItem,
  allItems,
  getId,
  onClose,
  onSelect,
}: UseItemNavigationProps<T>) {
  const selectNext = useCallback(() => {
    if (!currentItem || !allItems.length || !onSelect) return;
    const idx = allItems.findIndex((i) => getId(i) === getId(currentItem));
    if (idx < allItems.length - 1) onSelect(allItems[idx + 1]);
  }, [currentItem, allItems, getId, onSelect]);

  const selectPrev = useCallback(() => {
    if (!currentItem || !allItems.length || !onSelect) return;
    const idx = allItems.findIndex((i) => getId(i) === getId(currentItem));
    if (idx > 0) onSelect(allItems[idx - 1]);
  }, [currentItem, allItems, getId, onSelect]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (!isOpen) return;
      if (e.key === "Escape") { onClose(); return; }
      if (e.key === "j" || e.key === "J") { selectNext(); return; }
      if (e.key === "k" || e.key === "K") { selectPrev(); return; }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, selectNext, selectPrev, onClose]);

  return { selectNext, selectPrev };
}
