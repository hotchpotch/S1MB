"use client";
export default function ErrorPage() {
  return (
    <main className="max-w-3xl mx-auto p-8">
      <h1 className="text-2xl font-semibold">Results could not be loaded</h1>
      <p className="mt-3 text-muted-foreground">
        Check the data and result directories, then restart the viewer.
      </p>
    </main>
  );
}
