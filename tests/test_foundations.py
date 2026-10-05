"""Foundations: packaging, the domain-agnostic boundary, git safeguards, no AI attribution."""

import json
import re
import tomllib
from pathlib import Path

import judgekit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "judgekit"

# Terms that would mean domain knowledge leaked into the library.
DOMAIN_TERMS = (
    "insurance", "insurer", "deductible", "adjuster", "adjudicat", "policyholder",
    "airline", "flight", "passenger", "baggage", "tariff", "customs", "harmonized",
)  # fmt: skip
# Provider SDKs JudgeKit must never import: consumers adapt their own clients.
PROVIDER_MODULES = ("openai", "anthropic", "google", "groq", "httpx", "requests", "litellm")


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_version_matches_pyproject() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert judgekit.__version__ == meta["project"]["version"]


def test_runtime_dependencies_stay_minimal() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[<>=!~ \[]", d)[0].lower() for d in meta["project"]["dependencies"]}
    assert names <= {"pydantic", "pyyaml"}, f"new runtime dependency needs the user's OK: {names}"


def test_package_is_typed() -> None:
    assert (SRC / "py.typed").exists()


def test_no_domain_terms_in_library_source() -> None:
    hits = [
        f"{p.relative_to(ROOT)}: {term}"
        for p in _sources()
        for term in DOMAIN_TERMS
        if term in p.read_text(encoding="utf-8").lower()
    ]
    assert not hits, f"domain-specific terms in JudgeKit: {hits}"


def test_no_llm_provider_imports() -> None:
    pattern = re.compile(rf"^\s*(?:import|from)\s+({'|'.join(PROVIDER_MODULES)})\b", re.MULTILINE)
    hits = [str(p.relative_to(ROOT)) for p in _sources() if pattern.search(p.read_text("utf-8"))]
    assert not hits, f"provider imports in JudgeKit: {hits}"


def test_claude_settings_disable_attribution_and_block_history_rewrites() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["includeCoAuthoredBy"] is False
    assert settings["attribution"] == {"commit": "", "pr": ""}
    deny = settings["permissions"]["deny"]
    for verb in ("commit", "push", "rebase", "reset"):
        assert f"Bash(git {verb}:*)" in deny


def test_no_ai_attribution_in_repo_text() -> None:
    banned = re.compile(r"co-authored-by|generated with claude", re.IGNORECASE)
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md", ".toml", ".yml", ".yaml"}
    ]
    skip = {".venv", ".git", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    hits = [
        str(p.relative_to(ROOT))
        for p in files
        if not skip & set(p.relative_to(ROOT).parts)
        and banned.search(p.read_text("utf-8"))
        and p.name != "test_foundations.py"
    ]
    assert not hits, f"AI attribution found: {hits}"


def test_gitignore_covers_secrets_and_local_settings() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", ".venv/", ".claude/settings.local.json"):
        assert entry in text
