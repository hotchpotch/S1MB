import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'S1MB · System One Mosaic Benchmark', description: 'Fixed-sample evaluation of Choice, Noul, and Score models.', icons: { icon: '/favicon.svg' } };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
