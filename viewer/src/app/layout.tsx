import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'S1MB · System One Mosaic Benchmark', description: 'Fixed-sample evaluation of Choice, Noul, and Score models.', icons: { icon: '/favicon.svg' } };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><head><link rel="preload" href="/api/display/" as="fetch" crossOrigin="anonymous" /></head><body>{children}</body></html>;
}
