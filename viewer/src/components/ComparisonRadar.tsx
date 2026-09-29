"use client";

import { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowUp, Download, Link, X, Blocks, CircleCheck, ListChecks, BarChart3 } from 'lucide-react';
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
  let cursor = 128;
  const legend = profiles.map(profile => {
    const nameLines = lines(profile.name, 47);
    let repository = '';
    try {
      const url = new URL(profile.model.hf_url || profile.model.url || '');
      if (['huggingface.co', 'www.huggingface.co'].includes(url.hostname)) repository = decodeURI(url.pathname).split('/').filter(Boolean).slice(0, 2).join('/');
    } catch { /* Models without a valid repository URL omit the secondary label. */ }
    const repositoryLines = lines(repository, 76);
    const y = cursor;
    const cardHeight = nameLines.length * 24 + repositoryLines.length * 17 + 111;
    cursor += cardHeight + 12;
    return { profile, nameLines, repositoryLines, cardHeight, y };
  });
  const sourceLines = snapshot.resultsSource ? lines(`https://huggingface.co/datasets/${snapshot.resultsSource.repoId}`, 140) : [];
  const footerTop = Math.max(545, cursor + 8);
  const height = footerTop + (snapshot.resultsSource ? 65 + sourceLines.length * 18 : 38);

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
        <div className="overflow-x-auto rounded-lg border bg-card" tabIndex={0} aria-label="Scrollable chart preview">
          <svg ref={svg} xmlns="http://www.w3.org/2000/svg" width="1200" height={height} viewBox={`0 0 1200 ${height}`} role="img" aria-label="Six-axis model score comparison" className="w-full min-w-[560px] h-auto" style={{ fontFamily: 'Arial, sans-serif' }}>
            <title>{`S1MB model comparison · ${category.name}`}</title>
            <desc>{profiles.map(p => `${p.name}: ${RADAR_AXES.map((axis, i) => `${axis} ${score(p.values[i])}`).join(', ')}`).join('; ')}</desc>
            <rect width="1200" height={height} fill="#f5f8f5" />
            <rect width="1200" height="94" fill="#183e30" />
            <rect x="26" y="23" width="48" height="48" rx="12" fill="#2d5844" />
            <Blocks x={37} y={34} size={26} color="#d5eadb" strokeWidth={1.7} />
            <text x="90" y="43" fontSize="24" fontWeight="700" fill="#ffffff">S1MB <tspan fontWeight="400" fill="#d5eadb">/ Model comparison</tspan></text>
            <text x="90" y="68" fontSize="14" fill="#c0d5c7">{category.name}</text>
            <text x="1170" y="43" textAnchor="end" fontSize="12" fontWeight="700" letterSpacing="1.3" fill="#c0d5c7">ADJUSTED SCORES</text>
            <text x="1170" y="69" textAnchor="end" fontSize="18" fontWeight="600" fill="#ffffff">0–100 ↑</text>
            <g transform="translate(0 28)">
            {[100, 75, 50, 25].map(value => <polygon key={value} points={points(Array(6).fill(value))} fill={value === 100 ? '#ffffff' : 'none'} stroke="#d4dfd6" strokeWidth="1" />)}
            {RADAR_AXES.map((axis, index) => {
              const [x, y] = radarPoint(index, 100);
              const [lx, ly] = radarPoint(index, 100, 185);
              return <g key={axis}><line x1="285" y1="285" x2={x} y2={y} stroke="#d4dfd6" />
                <text x={lx} y={ly + 5} textAnchor={index === 0 || index === 3 ? 'middle' : index < 3 ? 'start' : 'end'} fontSize="16" fontWeight="600" fill={['#287456', '#3069a1', '#80529a'][index % 3]}>
                  {axis.startsWith('General ') ? <><tspan x={lx} dy="-7" fontSize="12" fontWeight="400" fill="#64766b">General</tspan><tspan x={lx} dy="19">{axis.replace('General ', '')}</tspan></> : axis}
                </text>
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
            </g>
            {legend.map(({ profile, nameLines, repositoryLines, cardHeight, y }) => {
              const metricsY = y + nameLines.length * 24 + repositoryLines.length * 17 + 20;
              return <g key={profile.id}>
                <rect x="535" y={y - 18} width="639" height={cardHeight} rx="12" fill="#ffffff" stroke="#dce5de" />
                <rect x="535" y={y - 2} width="4" height={cardHeight - 32} rx="2" fill={profile.color} />
                <line x1="554" x2="575" y1={y + 4} y2={y + 4} stroke={profile.color} strokeWidth="3" strokeDasharray={profiles.indexOf(profile) % 3 === 1 ? '7 3' : profiles.indexOf(profile) % 3 === 2 ? '2 3' : undefined} />
                {nameLines.map((line, i) => <text key={i} x="585" y={y + 10 + i * 24} fontSize="19" fontWeight="700" fill="#203d2e">{line}</text>)}
                {repositoryLines.map((line, i) => <text key={i} x="585" y={y + nameLines.length * 24 + 5 + i * 17} fontSize="12" fill="#748278">{line}</text>)}
                {RADAR_AXES.map((axis, i) => {
                  const Icon = [CircleCheck, ListChecks, BarChart3][i % 3];
                  const x = 554 + i * 102;
                  return <g key={axis}>
                    <Icon x={x} y={metricsY} size={14} color={['#287456', '#3069a1', '#80529a'][i % 3]} strokeWidth={1.8} />
                    <text x={x + 20} y={metricsY + 11} fontSize="11" fill="#64766b">
                      {axis.startsWith('General ') ? <><tspan x={x + 20}>General</tspan><tspan x={x + 20} dy="15">{axis.replace('General ', '')}</tspan></> : axis}
                    </text>
                    <text x={x} y={metricsY + 58} fontSize="27" fontWeight="700" fill={profile.color} style={{ fontVariantNumeric: 'tabular-nums' }}>{score(profile.values[i])}</text>
                  </g>;
                })}
              </g>;
            })}
            <line x1="28" x2="1172" y1={footerTop - 12} y2={footerTop - 12} stroke="#dce5de" />
            {snapshot.resultsSource && <text x="30" y={footerTop + 12} fontSize="12" fill="#64748b">Results commit: {snapshot.resultsSource.revision}{snapshot.resultsSource.refreshFailed ? ' · Cached; refresh failed' : ''}</text>}
            {sourceLines.map((line, index) => <text key={index} x="30" y={footerTop + 31 + index * 18} fontSize="12" fill="#64748b">{line}</text>)}
            <text x="30" y={height - 20} fontSize="12" fill="#64766b">System One Mosaic Benchmark{profiles.some(p => p.demo) ? ' · SYNTHETIC / DEMO' : ''}</text>
          </svg>
        </div>
        {message && <p role="status" className="text-sm text-muted-foreground">{message}</p>}
        {shareUrl && <label className="block text-xs text-muted-foreground">Comparison URL<input aria-label="Comparison URL" className="mt-1 w-full rounded border bg-background p-2 text-sm" readOnly value={shareUrl} onFocus={event => event.target.select()} /></label>}
      </div>
    </div>
  </section>;
}
