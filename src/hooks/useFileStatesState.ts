"use client";

import { useState, useEffect, type Dispatch, type SetStateAction } from "react";
import type { FileState, FileStatus } from "@/types";
import { fileKey } from "./useFilesRef";

interface StoredFileMeta {
  name: string;
  size: number;
  type: string;
  lastModified: number;
  status: FileStatus;
  error?: string;
  truncated?: boolean;
  charCount?: number;
  records?: Record<string, unknown>[];
}

/**
 * Like useSessionState but for FileState[] — serializes File objects as
 * metadata (File itself is non-serializable) and rehydrates by looking up
 * the real File in the provided module-level Map (see `useFilesRef`).
 *
 * If the real File is gone (e.g. after a full page reload), rehydrates
 * with an empty placeholder File so the row still renders.
 */
export function useFileStatesState(
  key: string,
  filesMap: Map<string, File>,
): [FileState[], Dispatch<SetStateAction<FileState[]>>] {
  const [value, setValue] = useState<FileState[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(key);
      if (saved !== null) {
        const metas = JSON.parse(saved) as StoredFileMeta[];
        const restored: FileState[] = metas.map((m) => {
          const lookupKey = `${m.name}__${m.size}`;
          let file = filesMap.get(lookupKey);
          if (!file) {
            file = new File([], m.name, { type: m.type, lastModified: m.lastModified });
            filesMap.set(fileKey(file), file);
          }
          return {
            file,
            status: m.status,
            error: m.error,
            truncated: m.truncated,
            charCount: m.charCount,
            records: m.records,
          };
        });
        setValue(restored);
      }
    } catch { /* parse error — keep empty */ }
    setHydrated(true);
  }, [key, filesMap]);

  useEffect(() => {
    if (!hydrated) return;
    try {
      const metas: StoredFileMeta[] = value.map((fs) => ({
        name: fs.file.name,
        size: fs.file.size,
        type: fs.file.type,
        lastModified: fs.file.lastModified,
        status: fs.status,
        error: fs.error,
        truncated: fs.truncated,
        charCount: fs.charCount,
        records: fs.records,
      }));
      sessionStorage.setItem(key, JSON.stringify(metas));
    } catch { /* quota exceeded — silently skip */ }
  }, [key, value, hydrated]);

  return [value, setValue];
}
