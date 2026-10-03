export type ParameterRange = [number, number];
export const ALL_PARAMETERS: ParameterRange = [0, 100];
const anchors = [50e6, 1e9, 5e9, 10e9, 20e9, 35e9];
export function parameterValue(position: number): number {
  const segment = Math.min(4, Math.floor(position / 20));
  return anchors[segment] + (anchors[segment + 1] - anchors[segment]) * (position / 20 - segment);
}
export function parameterLabel(position: number): string {
  if (position === 0) return '<50M';
  if (position === 100) return '35B+';
  const value = parameterValue(position);
  return value < 1e9 ? `${Math.round(value / 1e6)}M` : `${Number((value / 1e9).toFixed(2))}B`;
}
export function rangeActive(range: ParameterRange): boolean {
  return range[0] !== 0 || range[1] !== 100;
}
export function matchesParameterRange(value: number | null | undefined, range: ParameterRange): boolean {
  if (!rangeActive(range)) return true;
  return value != null && Number.isFinite(value) && value >= 0 &&
    (range[0] === 0 || value >= parameterValue(range[0])) &&
    (range[1] === 100 || value <= parameterValue(range[1]));
}
