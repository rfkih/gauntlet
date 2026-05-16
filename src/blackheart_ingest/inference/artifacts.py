"""Path-injected access to blackheart-train's content-addressed artifact
storage.

blackheart-train is not installed as a pip dependency in blackheart-ingest's
venv (the deployment model has them as sibling repos under C:\\Project,
not as inter-installed packages). For inference we need
:func:`blackheart_train.artifacts.read_artifact` to load the pickled
model payload by content_sha. The same path-inject pattern is used by
``tests/test_train_ingest_equivalence.py`` for the Session 1 feature
equivalence test.

When blackheart-train eventually moves to a pip-installable wheel
(future ops simplification), drop the path-inject and switch to a
plain ``from blackheart_train.artifacts import read_artifact``.

The injected path is computed once at import time and asserted to
exist. If the sibling repo isn't in the expected layout, the import
fails loudly rather than mysteriously at first use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# blackheart-ingest/src/blackheart_ingest/inference/artifacts.py
# -> parents[3] is blackheart-ingest/  ; parent is C:\Project\
_TRAIN_ROOT = _HERE.parents[3].parent / "blackheart-train"
_TRAIN_SRC = _TRAIN_ROOT / "src"
if not _TRAIN_SRC.exists():
    raise ImportError(
        f"Cannot path-inject blackheart-train: expected {_TRAIN_SRC} to exist. "
        f"Inference module expects sibling-repo layout (C:\\Project\\blackheart-train "
        f"alongside C:\\Project\\blackheart-ingest). Adjust path resolution if "
        f"the deployment layout has changed."
    )

if str(_TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SRC))

# blackheart-train's pydantic-settings reads ``.env`` from the *current
# working directory*, which when invoked from blackheart-ingest is the
# wrong location. The training loader needs TRAIN_DB_* to connect for
# feature_values reads, so eagerly load those vars from the sibling
# repo's .env file at import time. Only fills variables that aren't
# already set in the process environment — explicit env overrides
# always win.
_TRAIN_ENV_FILE = _TRAIN_ROOT / ".env"
if _TRAIN_ENV_FILE.exists():
    for line in _TRAIN_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

# Re-exported so callers don't need to know about the path-inject mechanism.
from blackheart_train.artifacts import (  # noqa: E402
    read_artifact as _read_artifact_inner,
)
from blackheart_train.specs import ModelSpec  # noqa: E402


# Default artifact directory. Mirrors blackheart-train's CLI default.
# Override via the ``artifact_dir`` parameter on :func:`load_artifact`
# for tests or alternative deployments.
DEFAULT_ARTIFACT_DIR = _HERE.parents[3].parent / "blackheart-train" / "artifacts"


def load_artifact(
    content_sha: str,
    *,
    artifact_dir: Path | None = None,
) -> dict:
    """Load and sha-verify a model artifact.

    Thin wrapper around blackheart_train.artifacts.read_artifact — the
    underlying function already verifies that payload['content_sha256']
    matches the filename's sha (tampering detection). We add nothing
    here except a default ``artifact_dir`` so the common inference call
    site doesn't have to plumb the path.

    Returns the full payload dict: spec, booster (or ensemble),
    feature_names, label_feature, metrics, deployment_readiness, etc.
    """
    return _read_artifact_inner(
        content_sha, artifact_dir or DEFAULT_ARTIFACT_DIR
    )


__all__ = ["load_artifact", "DEFAULT_ARTIFACT_DIR", "ModelSpec"]
