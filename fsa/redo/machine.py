"""Per-machine model endpoint profile: which Gemini endpoint every seat of the redo study calls.

The study's seats are Gemini models reached three ways: the native REST API (LiteLLM ``gemini/``
on the host: user simulator, tagger, claim book, ports; opencode's ``@ai-sdk/google`` inside the
trial containers: the agent), and the OpenAI-compatible endpoint (``openai`` SDK judges on the
host; the SWE-Interact simulator sidecar and rubric grader inside containers). A *profile* names
the base URLs of both APIs, as seen from the host and from inside Docker containers, and where
the API key comes from. Profiles live in ``configs/machines/profiles.json`` (committed).

Selection, first hit wins: ``TIB_MACHINE_PROFILE``; ``configs/machines/local.json``
``{"profile": "<name>"}`` (git-ignored, one per machine); the file's ``default`` (``gemini_api``:
the public Gemini API with ``GEMINI_API_KEY``, the original setup - nothing is overridden).

* ``apply_env(env)`` fills a subprocess environment for the active profile (keys, LiteLLM's
  ``GEMINI_API_BASE``, the benchmarks' agent base-URL switches, ``TIB_MACHINE_PROFILE`` so child
  processes resolve the same profile); ``apply_process_env()`` does it for this process.
* ``litellm_kwargs(model)`` / ``openai_base()`` / ``api_key()`` for direct calls.
* ``record()`` is the manifest record (never the key); ``model_record(id)`` the provider's model
  record through the active endpoint.
* ``host_arch()`` / ``check_results_root(dir)``: results trees are per architecture (a marker file
  refuses to mix runs of different hosts' architectures or profiles in one tree).
* ``drain_requested()`` / ``credentials_check()``: the pilots start no new trial while
  ``results/redo_v1/PILOT_DRAIN`` exists; the watchdog creates it while the optional machine-local
  check (``local.json`` ``credentials_check``, a shell command) fails, e.g. before the credentials
  behind a local endpoint expire, and removes it once the check passes again.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES_FILE = ROOT / "configs" / "machines" / "profiles.json"
LOCAL_FILE = ROOT / "configs" / "machines" / "local.json"
ENV_VAR = "TIB_MACHINE_PROFILE"
# Variables that only non-public profiles set (the public profile removes them, so a stray shell
# value can never silently redirect a "public" run).
BASE_VARS = ("GEMINI_API_BASE", "SWT_GOOGLE_BASE_URL", "GOOGLE_BASE_URL")
RESULTS_MARKER = "HOST.json"
# While this file exists the pilots start no new trial (running trials finish); the watchdog creates
# it before this machine's endpoint credentials expire and removes it once they are valid again.
DRAIN_FILE = ROOT / "results" / "redo_v1" / "PILOT_DRAIN"


def drain_requested() -> str | None:
    """The drain reason (first line of ``DRAIN_FILE``) while a drain is requested, else None."""
    try:
        return (DRAIN_FILE.read_text().strip().splitlines() or ["drain"])[0]
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class Profile:
    name: str
    source: str
    public: bool
    gemini_base: str
    openai_base: str
    container_gemini_base: str
    container_openai_base: str
    api_key_env: str
    api_key_file: str | None = None
    health_url: str | None = None
    description: str = ""

    def api_key(self) -> str | None:
        """The key: ``api_key_env`` when set, else the first line of ``api_key_file``."""
        key = os.environ.get(self.api_key_env)
        if key:
            return key.strip()
        if self.api_key_file:
            p = Path(os.path.expanduser(self.api_key_file))
            if p.is_file():
                return p.read_text().strip() or None
        return None

    def key_source(self) -> str:
        if os.environ.get(self.api_key_env):
            return f"env {self.api_key_env}"
        if self.api_key_file and Path(os.path.expanduser(self.api_key_file)).is_file():
            return f"file {self.api_key_file}"
        return "missing"


def profiles_doc() -> dict:
    return json.loads(PROFILES_FILE.read_text())


def local_doc() -> dict:
    """This machine's git-ignored ``configs/machines/local.json`` ({} when absent)."""
    return json.loads(LOCAL_FILE.read_text()) if LOCAL_FILE.is_file() else {}


def credentials_check(timeout: float = 120.0) -> dict:
    """Run this machine's optional endpoint-credentials check: ``local.json`` ``credentials_check``,
    a shell command that exits non-zero when the credentials behind a local endpoint expire soon
    (the watchdog drains the pilots while it fails). ``{"configured": bool, "ok": bool, ...}``."""
    cmd = str(local_doc().get("credentials_check") or "").strip()
    if not cmd:
        return {"configured": False, "ok": True}
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"configured": True, "ok": False, "rc": None, "error": f"timed out after {timeout:.0f}s"}
    return {"configured": True, "ok": r.returncode == 0, "rc": r.returncode}


def selected() -> tuple[str, str]:
    """(profile name, where the choice came from)."""
    name = os.environ.get(ENV_VAR, "").strip()
    if name:
        return name, f"env {ENV_VAR}"
    name = str(local_doc().get("profile") or "").strip()
    if name:
        return name, str(LOCAL_FILE.relative_to(ROOT))
    return profiles_doc()["default"], "default"


def get(name: str, source: str = "explicit") -> Profile:
    doc = profiles_doc()["profiles"]
    if name not in doc:
        raise SystemExit(f"unknown machine profile {name!r} (known: {sorted(doc)}; {PROFILES_FILE})")
    d = doc[name]
    return Profile(name=name, source=source, public=bool(d.get("public")),
                   gemini_base=d["gemini_base"].rstrip("/"), openai_base=d["openai_base"],
                   container_gemini_base=(d.get("container_gemini_base") or d["gemini_base"]).rstrip("/"),
                   container_openai_base=d.get("container_openai_base") or d["openai_base"],
                   api_key_env=d.get("api_key_env") or "GEMINI_API_KEY", api_key_file=d.get("api_key_file"),
                   health_url=d.get("health_url"), description=d.get("description", ""))


def active() -> Profile:
    name, source = selected()
    return get(name, source)


def api_key(profile: Profile | None = None) -> str | None:
    return (profile or active()).api_key()


def openai_base(profile: Profile | None = None) -> str:
    """OpenAI-compatible base URL (host side) - for the ``openai`` SDK seats."""
    return (profile or active()).openai_base


def gemini_base(profile: Profile | None = None) -> str:
    """Native REST base URL (host side), ending in ``/v1beta``."""
    return (profile or active()).gemini_base


def sidecar_litellm_base(profile: Profile | None = None) -> str:
    """``LITELLM_BASE_URL`` for SWE-Interact's simulator sidecar (in a container): the sidecar appends
    ``/v1`` and LiteLLM's ``openai/`` provider calls ``<base>/chat/completions``; the trailing ``#``
    turns the appended ``/v1`` into a URL fragment, which HTTP clients drop."""
    return (profile or active()).container_openai_base.rstrip("/") + "#"


def apply_env(env: dict, profile: Profile | None = None) -> dict:
    """Point a (subprocess) environment at the profile's endpoint; returns ``env``.

    Public profile: only ``TIB_MACHINE_PROFILE`` is set and the base-URL variables are removed;
    keys stay exactly as the caller set them (the original behaviour). Other profiles: the key
    replaces ``GEMINI_API_KEY`` / ``GOOGLE_GENERATIVE_AI_API_KEY`` (``GOOGLE_API_KEY`` is dropped:
    Harbor forwards the first Google key it finds), ``GEMINI_API_BASE`` points LiteLLM's ``gemini/``
    provider at the host-side base, and ``SWT_GOOGLE_BASE_URL`` (SWE-Together's patched harness) /
    ``GOOGLE_BASE_URL`` (Harbor 0.23.0, SWE-Interact) point the agents' opencode ``google`` provider
    at the container-side base."""
    p = profile or active()
    env[ENV_VAR] = p.name
    if p.public:
        for k in BASE_VARS:
            env.pop(k, None)
        return env
    key = p.api_key()
    if not key:
        raise RuntimeError(f"machine profile {p.name!r}: no API key (set {p.api_key_env}"
                           + (f" or create {p.api_key_file}" if p.api_key_file else "") + ")")
    env["GEMINI_API_KEY"] = key
    env["GOOGLE_GENERATIVE_AI_API_KEY"] = key
    env.pop("GOOGLE_API_KEY", None)
    env["GEMINI_API_BASE"] = p.gemini_base
    env["SWT_GOOGLE_BASE_URL"] = p.container_gemini_base
    env["GOOGLE_BASE_URL"] = p.container_gemini_base
    return env


def apply_process_env(profile: Profile | None = None) -> Profile:
    """``apply_env(os.environ)``: this process's LiteLLM calls and every child inherit the profile."""
    p = profile or active()
    apply_env(os.environ, p)  # type: ignore[arg-type]
    return p


def litellm_kwargs(model: str, profile: Profile | None = None) -> dict:
    """Extra ``litellm.completion`` kwargs for a ``gemini/`` model (empty for the public profile)."""
    p = profile or active()
    if p.public or not model.startswith("gemini/"):
        return {}
    kw: dict = {"api_base": p.gemini_base, "api_key": p.api_key()}
    if not model.startswith(("gemini/gemini-2.5", "gemini/gemini-3")):
        kw["allowed_openai_params"] = ["reasoning_effort"]
    return kw


def health(profile: Profile | None = None, timeout: float = 5.0) -> dict:
    """``{"ok": bool, ...}`` of the profile's health URL (public profiles: always ok, not probed)."""
    p = profile or active()
    if p.public or not p.health_url:
        return {"ok": True, "probed": False}
    try:
        with urllib.request.urlopen(p.health_url, timeout=timeout) as r:
            d = json.loads(r.read().decode() or "{}")
        return {"ok": bool(d.get("ok", True)), "probed": True}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "probed": True, "error": f"{type(e).__name__}: {e}"[:200]}


def require_healthy(profile: Profile | None = None, wait_sec: float = 0.0) -> None:
    """Refuse to start work while a local endpoint is down (optionally waiting ``wait_sec``)."""
    p = profile or active()
    deadline = time.monotonic() + wait_sec
    while True:
        h = health(p)
        if h["ok"]:
            return
        if time.monotonic() >= deadline:
            raise SystemExit(f"machine profile {p.name!r}: endpoint {p.health_url} is not healthy: "
                             f"{h.get('error')}")
        time.sleep(min(30.0, max(1.0, deadline - time.monotonic())))


def _sha256(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def host_arch() -> str:
    """Docker-style architecture of this host: ``amd64`` / ``arm64`` (else ``platform.machine()``)."""
    m = platform.machine().lower()
    return {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(m, m)


def record(profile: Profile | None = None) -> dict:
    """Manifest record of the endpoint profile in force (no secret)."""
    p = profile or active()
    return {"profile": p.name, "source": p.source, "public": p.public,
            "gemini_base": p.gemini_base, "openai_base": p.openai_base,
            "container_gemini_base": p.container_gemini_base, "container_openai_base": p.container_openai_base,
            "key": p.key_source(), "profiles_file": str(PROFILES_FILE.relative_to(ROOT)),
            "profiles_file_sha256": _sha256(PROFILES_FILE), "host_arch": host_arch()}


def model_record(model_id: str, profile: Profile | None = None) -> dict:
    """The provider's record of a Gemini model id through the active endpoint (``models.get``)."""
    p = profile or active()
    model_id = model_id.split("/", 1)[-1]
    try:
        req = urllib.request.Request(f"{p.gemini_base}/models/{model_id}",
                                     headers={"x-goog-api-key": p.api_key() or ""})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        rec = {k: d.get(k) for k in ("name", "version", "displayName", "inputTokenLimit",
                                     "outputTokenLimit", "thinking")}
        rec["endpoint_profile"] = p.name
        return rec
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200], "endpoint_profile": p.name}


def check_results_root(results: Path, profile: Profile | None = None) -> dict:
    """Results trees are per host architecture and endpoint profile: the first run writes
    ``<results>/HOST.json``; a later run on another architecture or profile is refused (runs of
    different arm64/amd64 images or providers must never share a tree). Returns the marker."""
    p = profile or active()
    marker = results / RESULTS_MARKER
    want = {"host_arch": host_arch(), "profile": p.name}
    if marker.is_file():
        have = json.loads(marker.read_text())
        diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
        if diff:
            raise SystemExit(f"{results} holds runs of another host/provider {diff} (marker {marker}); "
                             "use a separate results tree - never mix architectures or providers")
        return have
    results.mkdir(parents=True, exist_ok=True)
    doc = {**want, "created_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
           "note": "every run in this tree used this host architecture and endpoint profile"}
    marker.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Show / check the active machine endpoint profile.")
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "health", "env", "creds", "public"])
    a = ap.parse_args(argv)
    p = active()
    if a.cmd == "health":
        h = health(p)
        print(json.dumps(h))
        return 0 if h["ok"] else 1
    if a.cmd == "creds":  # exit 1 while the machine-local credentials check fails
        c = credentials_check()
        print(json.dumps(c))
        return 0 if c["ok"] else 1
    if a.cmd == "public":
        print(int(p.public))
        return 0
    if a.cmd == "env":  # shell-sourceable exports (the key itself is included: do not log this)
        env: dict = {}
        apply_env(env, p)
        for k in BASE_VARS:
            if k not in env:
                print(f"unset {k}")
        for k, v in env.items():
            print(f"export {k}={json.dumps(v)}")
        return 0
    print(json.dumps(record(p), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
