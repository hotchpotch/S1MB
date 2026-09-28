export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs' && process.env.S1MB_RESULTS_DIRS) {
    const { getSnapshot } = await import('./lib/results');
    await getSnapshot();
  }
}
