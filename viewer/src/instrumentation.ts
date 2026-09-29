export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs' && process.env.S1MB_DATA_DIR) {
    const { getSnapshot } = await import('./lib/results');
    const snapshot = await getSnapshot();
    console.log(`Loaded ${snapshot.results.length} validated results.`);
    for (const issue of snapshot.issues) console.warn(issue);
  }
}
