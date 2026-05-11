import type { NextConfig } from "next";

const isStaticExport = process.env.STATIC_BUILD === "1";
const computedBasePath = isStaticExport && process.env.PAGES_BASE_PATH ? process.env.PAGES_BASE_PATH : "";

const nextConfig: NextConfig = {
  // 'standalone' for web deployment / Docker (bundles server + minimal node_modules).
  // 'export' for static web (GitHub Pages) — no API routes.
  // Toggle via: STATIC_BUILD=1 npm run build:static
  output: isStaticExport ? "export" : "standalone",
  // GitHub Pages serves from a subpath (/repo-name/).
  ...(computedBasePath ? { basePath: computedBasePath, assetPrefix: computedBasePath } : {}),
  // Mirror basePath into a public env var so client-side `background-image: url(...)`
  // and other raw asset URLs can prefix correctly (Next only auto-prefixes <Image>/<Link>).
  env: {
    NEXT_PUBLIC_BASE_PATH: computedBasePath,
  },
  devIndicators: false,
  // pdf-parse v2 and pdfjs-dist v4 are ESM-only packages. They CANNOT be listed
  // in serverExternalPackages because that path uses require(), which fails on ESM.
  // Let Turbopack bundle them normally. The server-side code already disables the
  // pdfjs worker (workerSrc = "") so no worker resolution is needed at runtime.
};

export default nextConfig;
