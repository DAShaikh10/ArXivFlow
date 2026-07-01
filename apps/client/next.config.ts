import path from "node:path";
import type { NextConfig } from "next";

// The browser calls same-origin `/api/*`; the catch-all Route Handler at `src/app/api/[...path]`
// proxies those to the FastAPI backend (apps/api). The proxy target is read at request time from
// API_PROXY_TARGET, so it is a plain runtime env var (no rebuild to repoint). We intentionally do
// NOT use next.config `rewrites()` here: Next bakes rewrite destinations into the build, which would
// freeze the target at image-build time.
const config: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,

  sassOptions: {
    includePaths: [path.join(process.cwd(), "src/styles")],
    silenceDeprecations: ["legacy-js-api"],
  },

  experimental: {
    optimizePackageImports: ["react", "react-dom"],
  },
};

export default config;
