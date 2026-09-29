"use client";

import { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUp, Download, Link, X } from 'lucide-react';
import { Button } from './ui/button';
import { ModelName } from './ModelName';
import { RADAR_AXES, radarPoint, radarProfiles } from '../lib/radar';
import type { Category, Snapshot } from '../lib/types';

const score = (value: number | null) => value == null ? '—' : value.toFixed(1);
const points = (values: number[]) => values.map((value, axis) => radarPoint(axis, value).join(',')).join(' ');
function lines(text: string, size = 76): string[] {
  const output: string[] = [];
  let line = '', width = 0;
  for (const character of text) {
    const units = character.codePointAt(0)! > 255 ? 2 : /[MW]/.test(character) ? 1.5 : 1;
    if (width + units > size) { output.push(line); line = ''; width = 0; }
    line += character; width += units;
  }
  if (line) output.push(line);
  return output;
}

/** The displayed SVG is also the PNG source, so exports preserve the same scores and order. */
export function ComparisonRadar({ snapshot, category, selectedIds, onMove, onRemove }: {
  snapshot: Snapshot; category: Category; selectedIds: string[];
  onMove: (id: string, direction: -1 | 1) => void; onRemove: (id: string) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [shareUrl, setShareUrl] = useState('');
  const profiles = radarProfiles(snapshot, category, selectedIds);
  useEffect(() => { setShareUrl(''); setMessage(''); }, [selectedIds]);
  if (!profiles.length) return null;
  let cursor = 120;
  const legend = profiles.map(profile => {
    const nameLines = lines(profile.name, 58);
    const y = cursor;
    cursor += nameLines.length * 22 + 72;
    return { profile, nameLines, y };
  });
  const sourceLines = snapshot.resultsSource ? lines(`https://huggingface.co/datasets/${snapshot.resultsSource.repoId}`, 140) : [];
  const footerTop = Math.max(505, cursor + 12);
  const height = footerTop + (snapshot.resultsSource ? 120 + sourceLines.length * 18 : 88);

  async function download() {
    if (!svg.current) return;
    setBusy(true); setMessage('');
    let svgUrl: string | undefined, pngUrl: string | undefined;
    try {
      const scale = 2;
      if (height * scale > 16000) throw new Error('Select fewer models to fit the PNG size limit.');
      const source = new XMLSerializer().serializeToString(svg.current);
      svgUrl = URL.createObjectURL(new Blob([source], { type: 'image/svg+xml;charset=utf-8' }));
      const image = new Image();
      image.src = svgUrl;
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = 1200 * scale; canvas.height = height * scale;
      const context = canvas.getContext('2d');
      if (!context) throw new Error('Image export is unavailable in this browser.');
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob>((resolve, reject) => canvas.toBlob(value => value ? resolve(value) : reject(new Error('Could not create the PNG.')), 'image/png'));
      pngUrl = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = pngUrl; anchor.download = `s1mb-comparison-${category.id}.png`;
      document.body.append(anchor); anchor.click(); anchor.remove();
      // Give the browser time to start consuming the download before revoking it.
      const downloaded = pngUrl;
      setTimeout(() => URL.revokeObjectURL(downloaded), 60_000);
      pngUrl = undefined;
      setMessage('PNG downloaded.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Could not download the image.'); }
    finally { if (svgUrl) URL.revokeObjectURL(svgUrl); if (pngUrl) URL.revokeObjectURL(pngUrl); setBusy(false); }
  }
  async function copyLink() {
    const url = new URL(window.location.href);
    url.searchParams.set('view', 'compare'); url.searchParams.delete('run'); url.searchParams.delete('compare');
    for (const profile of profiles) url.searchParams.append('compare', profile.id);
    setShareUrl(url.href);
    try { await navigator.clipboard.writeText(url.href); setMessage('Comparison link copied.'); }
    catch { setMessage('Copy the comparison link below.'); }
  }

  return <section aria-label="Shareable comparison" className="space-y-3">
    <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.25fr)]">
      <div className="min-w-0 space-y-3">
        <h3 className="text-sm font-semibold">Selected models · Six adjusted scores</h3>
        <ol aria-label="Model comparison order" className="divide-y border-y">
          {profiles.map((profile, index) => <li key={profile.id} className="py-3 space-y-2">
            <div className="flex items-center gap-2">
              <span className="size-2.5 shrink-0 rounded-full" style={{ backgroundColor: profile.color }} />
              <div className="min-w-0 flex-1 text-sm"><ModelName model={profile.model} /></div>
              <Button variant="ghost" size="icon" className="size-7 shrink-0" aria-label={`Move ${profile.name} up`} disabled={index === 0} onClick={() => onMove(profile.id, -1)}><ArrowUp className="size-3.5" /></Button>
              <Button variant="ghost" size="icon" className="size-7 shrink-0" aria-label={`Move ${profile.name} down`} disabled={index === profiles.length - 1} onClick={() => onMove(profile.id, 1)}><ArrowDown className="size-3.5" /></Button>
              <Button variant="ghost" size="icon" className="size-7 shrink-0" aria-label={`Remove ${profile.name} from chart`} onClick={() => onRemove(profile.id)}><X className="size-3.5" /></Button>
            </div>
            <dl className="grid grid-cols-6 gap-1 text-center">
              {RADAR_AXES.map((axis, i) => <div key={axis}><dt className="flex h-8 flex-col justify-end text-[10px] leading-3.5 text-muted-foreground">{axis.startsWith('General ') && <span>General</span>}<span>{axis.replace('General ', '')}</span></dt><dd className="font-mono text-sm tabular-nums">{score(profile.values[i])}</dd></div>)}
            </dl>
          </li>)}
        </ol>
        <p className="text-xs text-muted-foreground">0–100 · Higher is better · — means incomplete or unavailable</p>
      </div>
      <div className="min-w-0 space-y-3">
        <div className="flex flex-wrap justify-end gap-2"><Button variant="outline" size="sm" onClick={copyLink}><Link className="size-4" />Copy comparison link</Button><Button size="sm" onClick={download} disabled={busy}><Download className="size-4" />{busy ? 'Creating PNG…' : 'Download PNG'}</Button></div>
        <div className="overflow-x-auto rounded-lg border bg-slate-50" tabIndex={0} aria-label="Scrollable chart preview">
          <svg ref={svg} xmlns="http://www.w3.org/2000/svg" width="1200" height={height} viewBox={`0 0 1200 ${height}`} role="img" aria-label="Six-axis model score comparison" className="w-full min-w-[560px] h-auto" style={{ fontFamily: 'Arial, sans-serif' }}>
            <title>{`S1MB model comparison · ${category.name}`}</title>
            <desc>{profiles.map(p => `${p.name}: ${RADAR_AXES.map((axis, i) => `${axis} ${score(p.values[i])}`).join(', ')}`).join('; ')}</desc>
            <rect width="1200" height={height} fill="#f8fafc" />
            <rect x="28" y="25" width="4" height="40" rx="2" fill="#6366f1" />
            <text x="46" y="42" fontSize="23" fontWeight="700" fill="#0f172a">S1MB · Model comparison</text>
            <text x="46" y="66" fontSize="14" fill="#475569">{category.name} · Adjusted scores · 0–100</text>
            <line x1="520" y1="98" x2="520" y2={footerTop - 28} stroke="#e2e8f0" />
            {[100, 75, 50, 25].map(value => <polygon key={value} points={points(Array(6).fill(value))} fill={value === 100 ? '#ffffff' : 'none'} stroke="#cbd5e1" strokeWidth="1" />)}
            {RADAR_AXES.map((axis, index) => {
              const [x, y] = radarPoint(index, 100);
              const [lx, ly] = radarPoint(index, 100, 185);
              return <g key={axis}><line x1="285" y1="285" x2={x} y2={y} stroke="#cbd5e1" />
                <text x={lx} y={ly + 5} textAnchor={index === 0 || index === 3 ? 'middle' : index < 3 ? 'start' : 'end'} fontSize="16" fontWeight="600" fill="#334155">{axis.replace("General ", "G. ")}</text>
              </g>;
            })}
            {[25, 50, 75, 100].map(value => <text key={value} x="293" y={285 - value * 1.4 + 4} fontSize="11" fill="#64748b">{value}</text>)}
            {profiles.map((profile, index) => <g key={profile.id}>
              <title>{profile.name}</title>
              {profile.values.every(v => v != null) && <polygon points={points(profile.values as number[])} fill={profile.color} fillOpacity="0.08" stroke={profile.color} strokeWidth="2.5" strokeLinejoin="round" strokeDasharray={index % 3 === 1 ? '8 4' : index % 3 === 2 ? '3 4' : undefined} />}
              {profile.values.map((value, axis) => {
                if (value == null) return null;
                const [x, y] = radarPoint(axis, value), next = profile.values[(axis + 1) % 6];
                const [nx, ny] = radarPoint((axis + 1) % 6, next ?? 0);
                return <g key={axis}>
                  {profile.values.some(v => v == null) && next != null && <line x1={x} y1={y} x2={nx} y2={ny} stroke={profile.color} strokeWidth="2.5" strokeDasharray="5 4" />}
                  <circle cx={x} cy={y} r="3.5" fill={profile.color} stroke="#fff" strokeWidth="1.5"><title>{`${profile.name} · ${RADAR_AXES[axis]}: ${score(value)}`}</title></circle>
                </g>;
              })}
            </g>)}
            {legend.map(({ profile, nameLines, y }) => <g key={profile.id}>
              <circle cx="550" cy={y - 6} r="5" fill={profile.color} />
              {nameLines.map((line, i) => <text key={i} x="565" y={y + i * 22} fontSize="18" fontWeight="600" fill={profile.color}>{line}</text>)}
              {RADAR_AXES.map((axis, i) => <g key={axis}>
                <text x={565 + i * 100} y={y + nameLines.length * 22 + 5} fontSize="13" fill="#64748b">{axis.startsWith('General ') ? 'G. ' : ''}{axis.replace('General ', '')}</text>
                <text x={565 + i * 100} y={y + nameLines.length * 22 + 29} fontSize="20" fontWeight="600" fill={profile.color}>{score(profile.values[i])}</text>
              </g>)}
              <line x1="540" x2="1170" y1={y + nameLines.length * 22 + 50} y2={y + nameLines.length * 22 + 50} stroke="#e2e8f0" />
            </g>)}
            <text x="30" y={footerTop} fontSize="13" fill="#475569">Higher is better · 0 = at or below baseline · 100 = reference ceiling · — = incomplete or unavailable</text>
            <text x="30" y={footerTop + 22} fontSize="12" fill="#64748b">G. / General = general-purpose subset. Scores are not accuracy or evidence of unseen-task generalization.</text>
            {snapshot.resultsSource && <text x="30" y={footerTop + 46} fontSize="12" fill="#64748b">Results commit: {snapshot.resultsSource.revision}{snapshot.resultsSource.refreshFailed ? ' · Cached; refresh failed' : ''}</text>}
            {sourceLines.map((line, index) => <text key={index} x="30" y={footerTop + 65 + index * 18} fontSize="12" fill="#64748b">{line}</text>)}
            <text x="30" y={height - 20} fontSize="12" fill="#64748b">System One Mosaic Benchmark{profiles.some(p => p.demo) ? ' · SYNTHETIC / DEMO' : ''}</text>
          </svg>
        </div>
        {message && <p role="status" className="text-sm text-muted-foreground">{message}</p>}
        {shareUrl && <label className="block text-xs text-muted-foreground">Comparison URL<input aria-label="Comparison URL" className="mt-1 w-full rounded border bg-background p-2 text-sm" readOnly value={shareUrl} onFocus={event => event.target.select()} /></label>}
      </div>
    </div>
  </section>;
}
