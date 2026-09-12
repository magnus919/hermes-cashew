# Pinned Hermes integration lane

This lane verifies the provider against real Hermes modules in a separate
process. It must not run through the repository `tests/conftest.py`, which
injects a local `MemoryProvider` and `MemoryManager` substitute.

The lane is supported on Python 3.11 through 3.13, matching the pinned Hermes
source, on the POSIX platforms supported by this plugin. Install the host
dependencies before running it. The command
does not install packages, access credentials, or write the real `~/.hermes`.

Prepare the pinned host source from the public immutable archive:

```sh
set -eu
REV=990473a79c6b0396b0a648fdd85ee8f7a5c267d3
SHA256=6c8585bfcb3807b7f0c1038be080429e0465137634745b9e48e11b65a9653bda
DEST=/tmp/hermes-agent-$REV
ARCHIVE=/tmp/hermes-agent-$REV.tar.gz
mkdir -p "$(dirname "$DEST")"
curl -fL "https://github.com/NousResearch/hermes-agent/archive/$REV.tar.gz" -o "$ARCHIVE"
printf '%s  %s\n' "$SHA256" "$ARCHIVE" | shasum -a 256 -c -
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "$ARCHIVE" --strip-components=1 -C "$DEST"
cat > "$DEST/.hermes-source.json" <<EOF
{"revision":"$REV","archive_sha256":"$SHA256"}
EOF
```

Create a separate host environment from the pinned checkout before the
offline run. Hermes' frozen lock supplies its host dependencies; Cashew's
runtime artifacts are pinned explicitly. The source and the older PyPI release
both report version `1.2.1`, so reinstall the source and verify its provenance:

```sh
HERMES_TEST_ENV=/tmp/hermes-agent-$REV-venv
UV_PROJECT_ENVIRONMENT="$HERMES_TEST_ENV" uv sync --frozen --no-install-project --project "$DEST"
CASHEW_PIN='cashew-brain @ https://github.com/rajkripal/cashew/archive/dd57ef029cf9a6dce0b8145d335a55202dd1bac4.tar.gz#sha256=38d2cb085fc8970a285991fca5df6b44309324b80947bb816f738a9acaaf72ab'
uv pip install --python "$HERMES_TEST_ENV/bin/python" \
  --reinstall "$CASHEW_PIN" 'sqlite-vec==0.1.9'
"$HERMES_TEST_ENV/bin/python" scripts/verify-cashew-baseline.py
```

Run the lane with the interpreter that has Hermes and Cashew installed:

```sh
HERMES_PINNED_SOURCE=/tmp/hermes-agent-990473a79c6b0396b0a648fdd85ee8f7a5c267d3 \
  "$HERMES_TEST_ENV/bin/python" integration/run_pinned_host.py
```

The command runs both the flat `hermes plugins install` loader and the
bundled/development loader. A missing or mismatched host source is an explicit
setup failure. The test patches only the embedding model and external client
resolution; Hermes loader, `MemoryManager`, `MemoryProvider`, auxiliary task
routing, `cron.jobs`, and SQLite remain real. It proves loader and lifecycle
compatibility with the verified source pin, not native embedding-model health
or complete process-crash containment.

Cron subprocess execution is part of the default command. The current main
branch includes the flat-install import fix from issue #186, so both the flat
and development scenarios are expected to pass. A cron failure remains an
actionable integration failure rather than a skipped check. The development
scenario can still be run alone with `--scenario dev` when isolating loader
behavior.
