#!/usr/bin/env python3
"""Launch one catalog app instance from a Git-tracked spec, no UI.

Same idea as the labs' clearml_apply.py: the desired state of each piece lives
in a JSON file next to this script, and this script sends it to the apps API.

    python deploy/klique_apply.py deploy/middleware.json --dry-run
    python deploy/klique_apply.py deploy/middleware.json

Spec format:
    {"app": "<catalog app id>", "notes": "...", "launch_params": {<launch-form fields>}}

Values written as ${ENV:NAME} are filled from the environment at apply time, so
site-specific values (compute resources, routes, image names) stay out of Git.
Secrets do not belong in a spec at all: launch parameters are stored on the
task. Put keys and tokens in a vault.

--dry-run prints the resolved request and contacts no server. A real apply
needs Klique credentials (CLEARML_API_* or ~/clearml.conf).

NOTE: the apps.launch_instance call shape is the one used in the labs' deploy
step; verify it against the server version on first use.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

_PLACEHOLDER = re.compile(r"\$\{ENV:([A-Za-z0-9_]+)\}")


def resolve(value, missing: set):
    """Replace ${ENV:NAME} in every string of a nested structure."""
    if isinstance(value, dict):
        return {k: resolve(v, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, missing) for v in value]
    if isinstance(value, str):
        def sub(match):
            name = match.group(1)
            if name not in os.environ:
                missing.add(name)
                return match.group(0)
            return os.environ[name]
        return _PLACEHOLDER.sub(sub, value)
    return value


def load_spec(path: Path, allow_missing: bool = False) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    missing: set = set()
    params = resolve(spec["launch_params"], missing)
    if missing and not allow_missing:
        raise SystemExit("set these environment variables first: " + ", ".join(sorted(missing)))
    name = params.get("task_name") or params.get("session_name") or params.get("name")
    return {"app": spec["app"], "launch_params": params, "task_name": name, "unresolved": sorted(missing)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch a Klique catalog app from a JSON spec")
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    request = load_spec(args.spec, allow_missing=args.dry_run)
    unresolved = request.pop("unresolved")
    if args.dry_run:
        print(json.dumps(request, indent=2))
        if unresolved:
            print("unresolved placeholders:", ", ".join(unresolved), file=sys.stderr)
        return

    from clearml.backend_api import Session

    res = Session().send_request(service="apps", action="launch_instance", method="post",
                                 json=request)
    ok = getattr(res, "ok", False)
    print("apps.launch_instance ->", getattr(res, "status_code", "?"),
          getattr(res, "text", str(res))[:400])
    if not ok:  # never let a failed launch pass silently
        raise SystemExit("launch failed")
    print("launched", request["app"], "as", request["task_name"])


if __name__ == "__main__":
    main()
