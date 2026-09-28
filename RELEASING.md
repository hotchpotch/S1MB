# Preparing a source release

1. Run the checks in CONTRIBUTING.md, including the public configuration without
   downloaded datasets. Build the Python distributions and viewer.
2. Inspect the release file list. Do not include `evaluator/data/datasets`,
   `evaluator/data/results` (except `.gitkeep`), `evaluator/audits`, checkpoints,
   local reports, `.env`, or tool output. Synthetic test fixtures are source assets.
3. Review dependency and bundled third-party licenses. Project code is MIT;
   dataset and model distribution permissions remain separate.
4. Document the dataset access requirement accurately. The configured Hugging Face
   repository currently requires authorization; a code release does not make it public.
5. Create a source archive from the reviewed commit:

   ```sh
   git archive --format=tar.gz --prefix=S1MB/ --output=/path/to/S1MB-source.tar.gz HEAD
   ```

The source archive includes only the current tracked tree and no Git history.
Historical development commits in this working repository contain evaluation
reports and dataset metadata. Removing those files from the current tree does
not remove them from history. Publish the clean source archive or a new repository
initialized from it; do not publish the historical Git repository without a separate
history review. Rewriting existing history or making repositories/datasets public
requires an explicit release decision.

Creating an archive is not a release or upload. Choose a version and destination
before publishing, and verify the contents of the actual artifact being uploaded.
