import type { RefObject } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

interface UseResizableWidthOptions {
  initialWidth: number;
  minWidth: number;
  getMaxWidth: () => number;
  stopPropagation?: boolean;
}

function useResizableWidth({
  initialWidth,
  minWidth,
  getMaxWidth,
  stopPropagation = false,
}: UseResizableWidthOptions) {
  const [width, setWidth] = useState(initialWidth);
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);
  const dragStartXRef = useRef(0);
  const dragStartWidthRef = useRef(0);

  const handleDragStart = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      if (stopPropagation) {
        event.stopPropagation();
      }
      isDraggingRef.current = true;
      setIsDragging(true);
      dragStartXRef.current = event.clientX;
      dragStartWidthRef.current = width;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [stopPropagation, width],
  );

  useEffect(() => {
    const handleDragMove = (event: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const delta = dragStartXRef.current - event.clientX;
      const maxWidth = getMaxWidth();
      const nextWidth = Math.min(
        maxWidth,
        Math.max(minWidth, dragStartWidthRef.current + delta),
      );
      setWidth(nextWidth);
    };

    const handleDragEnd = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      setIsDragging(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", handleDragMove);
    document.addEventListener("mouseup", handleDragEnd);
    return () => {
      document.removeEventListener("mousemove", handleDragMove);
      document.removeEventListener("mouseup", handleDragEnd);
    };
  }, [getMaxWidth, minWidth]);

  return { width, isDragging, handleDragStart };
}

interface UseResizableHeightOptions {
  initialHeight: number;
  minHeight: number;
  getMaxHeight: () => number;
}

function useResizableHeight({
  initialHeight,
  minHeight,
  getMaxHeight,
}: UseResizableHeightOptions) {
  const [height, setHeight] = useState(initialHeight);
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);
  const dragStartYRef = useRef(0);
  const dragStartHeightRef = useRef(0);

  const clampHeight = useCallback(
    (nextHeight: number) => {
      const maxHeight = Math.max(minHeight, getMaxHeight());
      return Math.min(maxHeight, Math.max(minHeight, nextHeight));
    },
    [getMaxHeight, minHeight],
  );

  const handleDragStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    isDraggingRef.current = true;
    setIsDragging(true);
    dragStartYRef.current = event.clientY;
    dragStartHeightRef.current = height;
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  }, [height]);

  const nudgeHeight = useCallback(
    (delta: number) => {
      setHeight((currentHeight) => clampHeight(currentHeight + delta));
    },
    [clampHeight],
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "ArrowUp") {
        event.preventDefault();
        nudgeHeight(-48);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        nudgeHeight(48);
      }
    },
    [nudgeHeight],
  );

  useEffect(() => {
    setHeight((currentHeight) => clampHeight(currentHeight));
  }, [clampHeight]);

  useEffect(() => {
    const handleDragMove = (event: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const delta = event.clientY - dragStartYRef.current;
      setHeight(clampHeight(dragStartHeightRef.current + delta));
    };

    const handleDragEnd = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      setIsDragging(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", handleDragMove);
    document.addEventListener("mouseup", handleDragEnd);
    return () => {
      document.removeEventListener("mousemove", handleDragMove);
      document.removeEventListener("mouseup", handleDragEnd);
    };
  }, [clampHeight]);

  return { height, isDragging, handleDragStart, handleKeyDown };
}

interface UseSplitPaneWidthOptions {
  containerRef: RefObject<HTMLElement | null>;
  preferredRatio: number;
  minStartWidth: number;
  minEndWidth: number;
  dividerWidth?: number;
}

function useSplitPaneWidth({
  containerRef,
  preferredRatio,
  minStartWidth,
  minEndWidth,
  dividerWidth = 14,
}: UseSplitPaneWidthOptions) {
  const [width, setWidth] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);
  const hasUserAdjustedRef = useRef(false);

  const clampWidth = useCallback(
    (nextWidth: number) => {
      const containerWidth = containerRef.current?.clientWidth ?? 0;
      if (containerWidth <= dividerWidth) {
        return Math.max(minStartWidth, nextWidth);
      }
      const availableWidth = containerWidth - dividerWidth;
      const maxWidth = Math.max(
        minStartWidth,
        availableWidth - minEndWidth,
      );
      return Math.min(maxWidth, Math.max(minStartWidth, nextWidth));
    },
    [containerRef, dividerWidth, minEndWidth, minStartWidth],
  );

  useEffect(() => {
    const syncWidth = () => {
      const containerWidth = containerRef.current?.clientWidth ?? 0;
      if (containerWidth <= dividerWidth) {
        return;
      }
      const availableWidth = containerWidth - dividerWidth;
      const preferredWidth = availableWidth * preferredRatio;
      setWidth((currentWidth) => {
        if (hasUserAdjustedRef.current && currentWidth !== null) {
          return clampWidth(currentWidth);
        }
        return clampWidth(preferredWidth);
      });
    };

    syncWidth();
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(syncWidth);
    observer.observe(node);
    return () => observer.disconnect();
  }, [clampWidth, containerRef, dividerWidth, preferredRatio]);

  const updateWidthFromClientX = useCallback(
    (clientX: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      hasUserAdjustedRef.current = true;
      setWidth(clampWidth(clientX - rect.left - dividerWidth / 2));
    },
    [clampWidth, containerRef, dividerWidth],
  );

  const handleDragStart = useCallback((event: React.MouseEvent) => {
    event.preventDefault();
    isDraggingRef.current = true;
    setIsDragging(true);
    updateWidthFromClientX(event.clientX);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, [updateWidthFromClientX]);

  const nudgeWidth = useCallback(
    (delta: number) => {
      hasUserAdjustedRef.current = true;
      setWidth((currentWidth) => {
        const containerWidth = containerRef.current?.clientWidth ?? 0;
        const fallbackWidth =
          containerWidth > dividerWidth
            ? (containerWidth - dividerWidth) * preferredRatio
            : minStartWidth;
        return clampWidth((currentWidth ?? fallbackWidth) + delta);
      });
    },
    [clampWidth, containerRef, dividerWidth, minStartWidth, preferredRatio],
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        nudgeWidth(-48);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        nudgeWidth(48);
      }
    },
    [nudgeWidth],
  );

  useEffect(() => {
    const handleDragMove = (event: MouseEvent) => {
      if (!isDraggingRef.current) return;
      updateWidthFromClientX(event.clientX);
    };

    const handleDragEnd = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      setIsDragging(false);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", handleDragMove);
    document.addEventListener("mouseup", handleDragEnd);
    return () => {
      document.removeEventListener("mousemove", handleDragMove);
      document.removeEventListener("mouseup", handleDragEnd);
    };
  }, [updateWidthFromClientX]);

  return { width, isDragging, handleDragStart, handleKeyDown };
}

export { useResizableHeight, useResizableWidth, useSplitPaneWidth };
