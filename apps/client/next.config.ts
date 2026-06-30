import path from "node:path";
import type { NextConfig } from "next";

// The FastAPI backend (apps/api). The browser calls same-origin `/api/*`; we proxy it here so dev
// has no CORS round-trips. Override for a deployed backend with API_PROXY_TARGET.
const API_TARGET = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

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

  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_TARGET}/api/:path*` }];
  },
};

export default config;
