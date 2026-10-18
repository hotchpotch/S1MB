'use client';

import { preload } from 'react-dom';

/** Begin the display request while the browser loads the dashboard scripts. */
export function DisplayPreload() {
  preload('/api/display/', { as: 'fetch', crossOrigin: 'anonymous' });
  return null;
}
