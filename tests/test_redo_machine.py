"""Per-machine endpoint profiles (fsa/redo/machine.py) and their wiring into the runners' seat
environments and direct model calls; SWE-Interact's per-architecture layer 1 (amd64.py). No network.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from fsa.redo import machine

LOCAL = "http://172.17.0.1:8765"


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "local_endpoint")
    monkeypatch.setenv("TIB_ENDPOINT_KEY", "local-secret")
    return machine.active()


def test_selection_order(monkeypatch, tmp_path):
    assert machine.selected() == ("gemini_api", "env TIB_MACHINE_PROFILE")  # conftest
    monkeypatch.delenv("TIB_MACHINE_PROFILE")
    lf = tmp_path / "local.json"
    monkeypatch.setattr(machine, "LOCAL_FILE", lf)
    assert machine.selected() == (machine.profiles_doc()["default"], "default") == ("gemini_api", "default")
    monkeypatch.setattr(machine, "ROOT", tmp_path)
    lf.write_text(json.dumps({"profile": "local_endpoint"}))
    assert machine.selected() == ("local_endpoint", "local.json")
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "gemini_api")
    assert machine.selected()[0] == "gemini_api"
    with pytest.raises(SystemExit):
        machine.get("nope")


def test_public_profile_changes_nothing_but_the_marker(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    env = {"GEMINI_API_KEY": "gkey", "GOOGLE_API_KEY": "x", "GEMINI_API_BASE": "http://stray",
           "SWT_GOOGLE_BASE_URL": "http://stray", "GOOGLE_BASE_URL": "http://stray"}
    machine.apply_env(env)
    assert env == {"GEMINI_API_KEY": "gkey", "GOOGLE_API_KEY": "x", "TIB_MACHINE_PROFILE": "gemini_api"}
    assert machine.litellm_kwargs("gemini/gemini-3.6-flash") == {}
    assert machine.openai_base() == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert machine.sidecar_litellm_base() == "https://generativelanguage.googleapis.com/v1beta/openai#"
    assert machine.health() == {"ok": True, "probed": False}


def test_local_profile_env(local):
    env = {"GEMINI_API_KEY": "personal", "GOOGLE_API_KEY": "personal"}
    machine.apply_env(env)
    assert env["GEMINI_API_KEY"] == env["GOOGLE_GENERATIVE_AI_API_KEY"] == "local-secret"
    assert "GOOGLE_API_KEY" not in env and env["TIB_MACHINE_PROFILE"] == "local_endpoint"
    assert env["GEMINI_API_BASE"] == f"{LOCAL}/v1beta"
    assert env["SWT_GOOGLE_BASE_URL"] == env["GOOGLE_BASE_URL"] == f"{LOCAL}/v1beta"
    assert machine.litellm_kwargs("gemini/gemini-3.6-flash") == {"api_base": f"{LOCAL}/v1beta",
                                                                 "api_key": "local-secret"}
    assert machine.litellm_kwargs("openai/gpt-x") == {}
    # the sidecar appends /v1; after '#' it is a URL fragment, never sent
    assert (machine.sidecar_litellm_base() + "/v1").split("#")[0] == f"{LOCAL}/v1beta/openai"


def test_local_profile_key_file_and_missing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "local_endpoint")
    monkeypatch.delenv("TIB_ENDPOINT_KEY", raising=False)
    p = machine.active()
    kf = tmp_path / "token"
    kf.write_text("from-file\n")
    p = machine.Profile(**{**p.__dict__, "api_key_file": str(kf)})
    assert p.api_key() == "from-file" and p.key_source() == f"file {kf}"
    p = machine.Profile(**{**p.__dict__, "api_key_file": str(tmp_path / "absent")})
    assert p.api_key() is None and p.key_source() == "missing"
    with pytest.raises(RuntimeError):
        machine.apply_env({}, p)


def test_record_has_no_secret(local):
    rec = machine.record()
    assert rec["profile"] == "local_endpoint" and rec["key"] == "env TIB_ENDPOINT_KEY"
    assert "local-secret" not in json.dumps(rec)
    assert rec["host_arch"] in ("amd64", "arm64") or rec["host_arch"]


def test_results_marker_refuses_mixing(local, tmp_path, monkeypatch):
    doc = machine.check_results_root(tmp_path)
    assert doc["profile"] == "local_endpoint" and (tmp_path / "HOST.json").is_file()
    assert machine.check_results_root(tmp_path)["host_arch"] == doc["host_arch"]
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "gemini_api")
    with pytest.raises(SystemExit):
        machine.check_results_root(tmp_path)
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "local_endpoint")
    monkeypatch.setattr(machine, "host_arch", lambda: "arm64" if doc["host_arch"] != "arm64" else "amd64")
    with pytest.raises(SystemExit):
        machine.check_results_root(tmp_path)


def test_env_cli_unsets_base_vars_for_public(capsys):
    assert machine.main(["env"]) == 0
    out = capsys.readouterr().out
    assert "unset GEMINI_API_BASE" in out and "export TIB_MACHINE_PROFILE=\"gemini_api\"" in out


def test_swe_together_seat_env_local_profile(local, monkeypatch):
    from fsa.redo.swe_together import run as swt_run
    from fsa.redo.swe_together.config import load_rungs
    monkeypatch.setenv("GEMINI_API_KEY", "personal")
    monkeypatch.setattr(machine, "host_arch", lambda: "amd64")
    cfg = load_rungs()
    env = swt_run.seat_env(cfg, "docker")
    assert env["GEMINI_API_KEY"] == env["GOOGLE_GENERATIVE_AI_API_KEY"] == "local-secret"
    assert env["GEMINI_API_BASE"] == f"{LOCAL}/v1beta" and env["SWT_GOOGLE_BASE_URL"] == f"{LOCAL}/v1beta"
    assert swt_run.sandbox_key("docker") == "docker_amd64" and swt_run.sandbox_key("e2b") == "e2b"
    assert "SWT_DOCKER_IMAGE_TEMPLATE" not in env  # official (native amd64) task images
    assert env["TRIAL_BUDGET_SEC"] == str(cfg["sandboxes"]["docker_amd64"]["trial_budget_sec"])


def test_swe_together_seat_env_arm64_keeps_template(monkeypatch):
    from fsa.redo.swe_together import run as swt_run
    from fsa.redo.swe_together.config import load_rungs
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    monkeypatch.setattr(machine, "host_arch", lambda: "arm64")
    cfg = load_rungs()
    env = swt_run.seat_env(cfg, "docker")
    assert env["SWT_DOCKER_IMAGE_TEMPLATE"] == cfg["sandboxes"]["docker_arm64"]["image_template"]
    assert env["GEMINI_API_KEY"] == "gkey" and "GEMINI_API_BASE" not in env


def test_swe_interact_seat_env_local_profile(local, monkeypatch):
    from fsa.redo.swe_interact import run as swi_run
    from fsa.redo.swe_interact.config import TASKS, load_rungs, subset_tasks
    if not (TASKS.is_dir() and (TASKS.parents[1] / ".git").exists()):
        pytest.skip("no SWE-Interact checkout")
    monkeypatch.setenv("GEMINI_API_KEY", "personal")
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key")
    env = swi_run.seat_env(load_rungs(), subset_tasks()[0])
    assert env["LITELLM_BASE_URL"] == f"{LOCAL}/v1beta/openai#"
    assert env["OPENAI_API_BASE"] == f"{LOCAL}/v1beta/openai/"
    assert env["LITELLM_API_KEY"] == env["OPENAI_API_KEY"] == env["GOOGLE_GENERATIVE_AI_API_KEY"] == "local-secret"
    assert env["GOOGLE_BASE_URL"] == f"{LOCAL}/v1beta" and env["GEMINI_API_BASE"] == f"{LOCAL}/v1beta"
    assert "GEMINI_API_KEY" not in env and "GOOGLE_API_KEY" not in env
    assert "real-openai-key" not in env.values() and "personal" not in env.values()
    pub = swi_run.public_env(env)
    assert pub["GOOGLE_BASE_URL"] == f"{LOCAL}/v1beta" and "local-secret" not in json.dumps(pub)


def test_direct_litellm_calls_take_the_profile(local, monkeypatch):
    import os
    import sys
    import types
    seen = []
    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=lambda **kw: seen.append(kw)))
    from fsa.redo.swe_interact.arms import RedoConfig, SwiArms
    SwiArms(RedoConfig(arm="commit_before"), "/tmp", "google/gemini-3.6-flash")._complete(
        model="gemini/gemini-3.6-flash", messages=[])
    assert [(k["api_base"], k["api_key"]) for k in seen] == [(f"{LOCAL}/v1beta", "local-secret")]
    # fsa/redo/ports.py must stay byte-identical to the SWE-Together clone copy: its LiteLLM calls
    # get the endpoint from the process environment (LiteLLM's gemini/ provider reads both)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_BASE", "GOOGLE_GENERATIVE_AI_API_KEY", "SWT_GOOGLE_BASE_URL",
              "GOOGLE_BASE_URL"):
        monkeypatch.setenv(k, "")  # recorded, so teardown also removes what apply_process_env adds
        monkeypatch.delenv(k)
    machine.apply_process_env()
    assert os.environ["GEMINI_API_BASE"] == f"{LOCAL}/v1beta" and os.environ["GEMINI_API_KEY"] == "local-secret"


def test_llm_client_follows_profile(local, tmp_path):
    from fsa.llm.client import LLM
    llm = LLM("gemini-3.6-flash", cache_path=str(tmp_path / "c.sqlite"), use_cache=False)
    assert str(llm.client.base_url).rstrip("/") == f"{LOCAL}/v1beta/openai" and llm.endpoint_profile == "local_endpoint"
    assert llm.client.api_key == "local-secret"


def test_llm_client_public_unchanged(monkeypatch, tmp_path):
    from fsa.llm.client import LLM
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    llm = LLM("gemini-3.6-flash", cache_path=str(tmp_path / "c.sqlite"), use_cache=False)
    assert str(llm.client.base_url) == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert llm.endpoint_profile is None and llm.client.api_key == "gkey"


def test_drain_requested(monkeypatch, tmp_path):
    f = tmp_path / "PILOT_DRAIN"
    monkeypatch.setattr(machine, "DRAIN_FILE", f)
    assert machine.drain_requested() is None
    f.write_text("credentials expire in < 100 min\nsince 2026-09-29T12:57Z\n")
    assert machine.drain_requested() == "credentials expire in < 100 min"
    f.write_text("")
    assert machine.drain_requested() == "drain"


def test_credentials_check(monkeypatch, tmp_path):
    lf = tmp_path / "local.json"
    monkeypatch.setattr(machine, "LOCAL_FILE", lf)
    assert machine.credentials_check() == {"configured": False, "ok": True}
    lf.write_text(json.dumps({"profile": "local_endpoint", "credentials_check": "exit 3"}))
    assert machine.credentials_check() == {"configured": True, "ok": False, "rc": 3}
    lf.write_text(json.dumps({"credentials_check": "true"}))
    assert machine.credentials_check()["ok"] is True
    assert machine.main(["creds"]) == 0


def test_swe_interact_layer1_follows_host_arch(monkeypatch):
    from fsa.redo.swe_interact import patchset
    assert patchset.layer1("amd64") == patchset.AMD64 and patchset.layer1("arm64") == patchset.ARM64
    monkeypatch.setattr(machine, "host_arch", lambda: "amd64")
    assert patchset.layer1() == patchset.AMD64
    monkeypatch.setattr(machine, "host_arch", lambda: "arm64")
    assert patchset.layer1() == patchset.ARM64


def test_swe_interact_amd64_image_tag():
    from fsa.redo.swe_interact import amd64
    assert amd64.image_tag("rf_task-1") == "tib/swe-interact-amd64:rf_task-1"
    exact = "x" * 128
    assert amd64.image_tag(exact) == f"tib/swe-interact-amd64:{exact}"
    long = "swebenchpro_instance_" + "a" * 120
    tag = amd64.image_tag(long).split(":", 1)[1]
    assert len(tag) == 128 and tag.startswith(long[:119] + "-") and tag != amd64.image_tag(long + "b").split(":", 1)[1]


def test_swe_interact_amd64_retag():
    from fsa.redo.swe_interact import amd64
    src = '[metadata]\ndocker_image = "keep"\n\n[environment]\ncpus = 2\ndocker_image = "old"\nmemory_mb = 8192\n'
    out = amd64.retag(src, "tib/x:t")
    assert out == src.replace('docker_image = "old"', 'docker_image = "tib/x:t"')
    src2 = '[environment]\ncpus = 2\n\n[verifier]\ntimeout_sec = 10\n'
    assert amd64.retag(src2, "tib/x:t") == '[environment]\ndocker_image = "tib/x:t"\ncpus = 2\n\n[verifier]\ntimeout_sec = 10\n'
    with pytest.raises(SystemExit):
        amd64.retag("[verifier]\ntimeout_sec = 10\n", "tib/x:t")


def _swi_tasks():
    from fsa.redo.swe_interact.config import TASKS, subset_tasks
    if not (TASKS.is_dir() and (TASKS.parents[1] / ".git").exists()):
        pytest.skip("no SWE-Interact checkout")
    return subset_tasks()


def test_swe_interact_amd64_recipes():
    from fsa.redo.swe_interact import amd64
    tasks = _swi_tasks()
    by_family = {f: [t for t in tasks if t.startswith(f)] for f in ("deepswe_", "rf_", "swebenchpro_")}
    assert all(by_family.values())
    for t in tasks:
        rc = amd64.recipe(t)
        lines = rc["dockerfile"].splitlines()
        froms = [ln.split()[1] for ln in lines if ln.upper().startswith("FROM ")]
        assert rc["base"] == froms[-1] and rc["dockerfile_sha256"] == hashlib.sha256(rc["dockerfile"].encode()).hexdigest()
        pub = amd64.published_image(t)
        if t.startswith("deepswe_"):  # published environment + the multi-turn additions only
            assert rc["kind"] == "published_image_plus_multiturn_additions" and froms == [pub]
            first = next(ln for ln in lines if ln and not ln.startswith(("#", "FROM ")))
            assert amd64.COPY_LINE.match(first)
        elif t.startswith("swebenchpro_"):  # built-based at the pin: its own Dockerfile verbatim
            assert pub is None and rc["kind"] in ("task_dockerfile", "task_dockerfile+harbor_deps_snapshot")
        else:  # rf: the task Dockerfile starts FROM the published image
            assert froms == [pub] and rc["kind"].startswith("task_dockerfile")
        pinned = t in amd64.APT_PIN_TASKS
        deps = t in amd64.HARBOR_DEPS_TASKS
        quota = t in amd64.NODE_CPU_QUOTA_TASKS
        assert rc["kind"].endswith("+node_cpu_quota") == quota
        end = lines.index(f"COPY <<'TIB_EOF' {amd64.NODE_CPU_QUOTA_JS}") - 2 if quota else len(lines)
        assert not pinned or t.startswith("rf_")  # the own apt step of Debian 11 rf images
        assert ("+bullseye_security_pin" in rc["kind"]) == pinned and ("+harbor_deps_snapshot" in rc["kind"]) == deps
        if pinned:
            pin_at = next(i for i, ln in enumerate(lines) if amd64.APT_PIN_FILE in ln and "printf" in ln)
            copy_at = next(i for i, ln in enumerate(lines) if amd64.COPY_LINE.match(ln))
            rm_at = lines.index(f"RUN rm -f {amd64.APT_PIN_FILE}")
            assert copy_at < pin_at < rm_at
            assert not any(ln.startswith("RUN ") for ln in lines[copy_at + 1:pin_at])
            rest = [ln for ln in lines[rm_at + 1:end] if ln.strip()]
            assert (rest == []) == (not deps)
        if deps:  # last (after any pin is gone): Harbor's installer packages from the security snapshot,
            # the image's own bullseye-security source swapped for it and restored after the install
            start = next(i for i, ln in enumerate(lines) if ln.startswith("RUN cp /etc/apt/sources.list /tmp/tib-sources.list"))
            assert not any(ln.startswith(("RUN ", "FROM ", "USER ")) for ln in lines[start + 1:end])
            body = "\n".join(lines[start:])
            assert f"sed -i -E 's#^deb {amd64.HARBOR_DEPS_TASKS[t]} #deb {amd64.SECURITY_SNAPSHOT} #'" in body
            assert f"apt-get install -y {amd64.HARBOR_DEPS}" in body
            assert "mv /tmp/tib-sources.list /etc/apt/sources.list" in body
        if quota:  # last: the Node CPU-quota preload, appended to the image's own NODE_OPTIONS
            at = end + 2
            assert lines[end].startswith("# tib redo_v1 amd64") and lines[end + 1].startswith("# ")
            close = lines.index("TIB_EOF", at)
            assert "\n".join(lines[at + 1:close]) + "\n" == amd64.NODE_CPU_QUOTA_SRC
            assert "/sys/fs/cgroup/cpu.max" in amd64.NODE_CPU_QUOTA_SRC and "availableParallelism" in amd64.NODE_CPU_QUOTA_SRC
            assert lines[close + 1] == f'ENV NODE_OPTIONS="${{NODE_OPTIONS}} --require {amd64.NODE_CPU_QUOTA_JS}"'
            assert lines[close + 2].startswith("RUN node -e") and len(lines) == close + 3


def test_swe_interact_amd64_patch_reproducible():
    from fsa.redo.swe_interact import amd64
    from fsa.redo.swe_interact.config import load_rungs
    cfg = load_rungs()["benchmark"]
    subset = _swi_tasks()
    spro = amd64.swebenchpro_tasks()
    tasks = amd64.patch_tasks()
    # A5.13 (1): layer 1 = Stage-1 subset (20) + every swebenchpro task at the pin (25), 7 overlapping
    assert len(subset) == 20 and len(spro) == 25 and all(t.startswith("swebenchpro_") for t in spro)
    assert tasks == sorted(set(subset) | set(spro)) and len(tasks) == 38
    assert len(set(subset) & set(spro)) == 7
    text = amd64.expected_patch()
    assert text == amd64.PATCH.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == cfg["amd64_patch_sha256"]
    assert text.count("diff --git") == len(tasks)
    assert all(f'+docker_image = "{amd64.image_tag(t)}"' in text for t in tasks)
    # the Stage-1 layer (subset only) is still reproduced byte for byte, sha recorded in rungs.json
    stage1 = amd64.expected_patch(subset)
    assert hashlib.sha256(stage1.encode()).hexdigest() == cfg["amd64_patch_stage1_sha256"]
    assert stage1.count("diff --git") == len(subset)
    assert cfg["amd64_patch_stage1_sha256"] != cfg["amd64_patch_sha256"]


def test_swe_interact_task_images():
    import re
    import subprocess
    from fsa.redo.swe_interact import amd64
    from fsa.redo.swe_interact.config import ROOT, TASKS, environment_content_hash, task_docker_image, task_image
    tasks = _swi_tasks()
    if all(task_docker_image(t) == amd64.image_tag(t) for t in tasks):  # amd64 layout (amd64.patch applied)
        assert all(task_image(t) == amd64.image_tag(t) for t in tasks)
    else:
        assert all(task_image(t) == (task_docker_image(t) or f"hb__{environment_content_hash(TASKS / t / 'environment')}")
                   for t in tasks)
    built = [t for t in tasks if amd64.published_image(t) is None]  # build-based at the pin (swebenchpro)
    assert built and all(re.fullmatch(r"[0-9a-f]{32}", environment_content_hash(TASKS / t / "environment")) for t in built)
    harbor_py = ROOT / "third_party" / "swe-interact" / ".venv" / "bin" / "python"
    if not harbor_py.exists():
        pytest.skip("no Harbor venv")
    code = ("import sys; from pathlib import Path; from harbor.environments.definition import environment_content_hash as h; "
            "print(' '.join(h(Path(p)) for p in sys.argv[1:]))")
    out = subprocess.run([str(harbor_py), "-c", code, *[str(TASKS / t / "environment") for t in built]],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == [environment_content_hash(TASKS / t / "environment") for t in built]  # Harbor's hb__ naming



# ── exploratory frontier rung plumbing (profile local_endpoint_all, G rung, timeout override) ──────

ALL = "http://127.0.0.1:8766"
ALL_CONTAINER = "http://172.17.0.1:8766"


def test_local_endpoint_all_profile_env(monkeypatch):
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "local_endpoint_all")
    monkeypatch.setenv("TIB_ENDPOINT_KEY", "local-secret")
    p = machine.active()
    assert not p.public and p.health_url == f"{ALL}/healthz"
    env: dict = {}
    machine.apply_env(env, p)
    assert env["GEMINI_API_BASE"] == f"{ALL}/v1beta"                       # host-side seats
    assert env["SWT_GOOGLE_BASE_URL"] == env["GOOGLE_BASE_URL"] == f"{ALL_CONTAINER}/v1beta"  # in-container seats
    assert (machine.sidecar_litellm_base(p) + "/v1").split("#")[0] == f"{ALL_CONTAINER}/v1beta/openai"
    # the pre-registered profile is untouched
    q = machine.get("local_endpoint")
    assert q.gemini_base == f"{LOCAL}/v1beta" and q.container_gemini_base == f"{LOCAL}/v1beta"


def test_llm_timeout_env_override(monkeypatch):
    from fsa.llm.client import LLM
    monkeypatch.setenv("TIB_MACHINE_PROFILE", "local_endpoint")
    monkeypatch.setenv("TIB_ENDPOINT_KEY", "local-secret")
    monkeypatch.delenv("FSA_LLM_TIMEOUT_SECONDS", raising=False)
    assert LLM("gemini-3.8-flash", use_cache=False, reasoning_effort="low").timeout == 120.0  # default unchanged
    monkeypatch.setenv("FSA_LLM_TIMEOUT_SECONDS", "900")
    assert LLM("gemini-3.8-flash", use_cache=False, reasoning_effort="low").timeout == 900.0


def test_swi_g_rung_and_per_rung_variant(monkeypatch, tmp_path):
    from fsa.redo.swe_interact import run as R
    cfg = R.load_rungs()
    assert {"W", "M", "S", "G"} <= set(cfg["rungs"]) <= {"W", "F", "M", "S", "G"}
    assert cfg["rungs"]["G"]["model"] == "google/gemini-4.0-argon"
    assert "exploratory" in cfg["rungs"]["G"]
    for r in ("W", "M", "S"):
        assert "variant" not in cfg["rungs"][r]  # pre-registered rungs keep the harness-wide variant
    s_cmd = R.harbor_cmd(cfg, "S", "deepswe_testem-bail-on-test-failure", tmp_path, "j")
    assert "variant=high" in s_cmd and "-m" in s_cmd and s_cmd[s_cmd.index("-m") + 1] == "google/gemini-3.8-flash"
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["rungs"]["G"]["variant"] = "low"
    g_cmd = R.harbor_cmd(cfg2, "G", "deepswe_testem-bail-on-test-failure", tmp_path, "j")
    assert "variant=low" in g_cmd and g_cmd[g_cmd.index("-m") + 1] == cfg["rungs"]["G"]["model"]


def test_swi_results_root_override(monkeypatch, tmp_path):
    import importlib
    from fsa.redo.swe_interact import config as C
    canonical = C.CANONICAL_RESULTS
    assert C.RESULTS == canonical  # default: the pre-registered tree
    monkeypatch.setenv("TIB_RESULTS_ROOT", str(tmp_path / "g4"))
    C2 = importlib.reload(C)
    try:
        assert C2.RESULTS == (tmp_path / "g4").resolve() and C2.CANONICAL_RESULTS == canonical
    finally:
        monkeypatch.delenv("TIB_RESULTS_ROOT")
        importlib.reload(C)
    assert C.RESULTS == canonical
