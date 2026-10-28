# Bundled upstream Cashew

This is a private dependency snapshot, not a local engine implementation.
`provenance.json` identifies the immutable upstream archive, commit, tree, and
original/runtime hashes. `LICENSE` preserves the upstream MIT notice.

`pristine-core.tar.gz` contains all 24 original `core` files and the original
`scripts/migrate_embeddings.py` used by the adapter. Runtime copies preserve
algorithms. The only packaging transformations are:

- In core files, `from core.X` becomes `from .X`.
- In the migration script, `from core.X` becomes `from ..core.X`.
- Remove that standalone script's `ROOT`/`sys.path` bootstrap; package imports
  resolve through the private namespace instead.

From the repository root, reproduce the snapshot using:

```sh
python scripts/vendor-cashew.py /path/to/fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz
python scripts/verify-cashew-baseline.py
```

The bundler validates the full archive SHA-256 before reading its files. The
offline verifier anchors the pristine snapshot digest and compares every runtime
file with the allowlisted transformation of its original. Formatting or editing
engine code fails this check; upstream behavior changes need a separately tested
source update. The adapter uses only private relative imports, so an installed
or pre-imported top-level `core` package does not select the engine.
