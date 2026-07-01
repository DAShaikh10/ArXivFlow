import type { Metadata } from "next";

import { Providers } from "./providers";
import "@/styles/main.scss";

export const metadata: Metadata = {
  title: "ArXivFlow",
  description:
    "ArXiv research paper recommender system - Semantic paper retrieval & embedding atlas over an indexed arXiv NLP corpus.",
};

const FONTS_HREF =
  "https://fonts.googleapis.com/css2?" +
  "family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&" +
  "family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&" +
  "family=IBM+Plex+Sans:wght@400;500;600;700&" +
  "family=IBM+Plex+Mono:wght@400;500;600&" +
  "family=Space+Grotesk:wght@400;500;600;700&" +
  "family=Spline+Sans:wght@400;500;600;700&display=swap";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="stylesheet" href={FONTS_HREF} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
