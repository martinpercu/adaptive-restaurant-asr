"""Build Whisper `initial_prompt` from the store menu (plan/02-architecture.md §3).

Biases decoding toward real menu terms (the restaurant prior). Capped at
`initial_prompt_max_tokens` (<= 224, Whisper truncates prompts there).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_menu(menu_dir: str | Path, store_id: str = "demo") -> dict:
    path = Path(menu_dir) / f"{store_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def menu_terms(menu: dict, lang: str) -> list[str]:
    """Ordered, de-duplicated domain terms for a language (names + aliases + service)."""
    terms: list[str] = []
    for item in menu.get("items", []):
        terms.append(item["name"][lang])
        terms.extend(item.get("aliases", {}).get(lang, []))
    for svc in menu.get("service_terms", []):
        terms.append(svc[lang])
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _cap(terms: list[str], max_tokens: int) -> str:
    chosen: list[str] = []
    budget = 0
    for term in terms:
        cost = len(term.split())
        if budget + cost > max_tokens:
            break
        chosen.append(term)
        budget += cost
    return ", ".join(chosen)


def build_initial_prompt(menu: dict, lang: str, max_tokens: int = 200) -> str:
    """Comma-joined menu terms for one language, truncated to ~max_tokens tokens."""
    return _cap(menu_terms(menu, lang), max_tokens)


def build_bilingual_prompt(menu: dict, max_tokens: int = 200) -> str:
    """Interleave es+en terms so the prompt biases toward the menu in either language.

    Language is detected inside the engine, so the per-request prompt covers both.
    """
    es, en = menu_terms(menu, "es"), menu_terms(menu, "en")
    interleaved: list[str] = []
    for a, b in zip(es, en, strict=False):
        interleaved.extend([a, b])
    interleaved.extend(es[len(en) :] or en[len(es) :])
    # de-dup preserving order
    seen: set[str] = set()
    uniq = [t for t in interleaved if not (t.lower() in seen or seen.add(t.lower()))]
    return _cap(uniq, max_tokens)
