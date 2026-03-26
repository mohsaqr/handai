"use client";

import { useCallback } from "react";
import { useProcessingStore } from "@/lib/processing-store";

/**
 * Lightweight hook for custom-loop tools to register processing status
 * in the global store, enabling sidebar processing indicators.
 *
 * Usage:
 *   const { markProcessing, markIdle } = useProcessingFlag("/my-tool");
 *   // call markProcessing() when processing starts
 *   // call markIdle() when processing ends
 */
export function useProcessingFlag(toolId: string) {
  const setProcessingFlag = useProcessingStore((s) => s.setProcessingFlag);

  const markProcessing = useCallback(() => {
    setProcessingFlag(toolId, true);
  }, [toolId, setProcessingFlag]);

  const markIdle = useCallback(() => {
    setProcessingFlag(toolId, false);
  }, [toolId, setProcessingFlag]);

  return { markProcessing, markIdle };
}
