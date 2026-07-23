"""Deterministic Mermaid node-to-evidence mapping."""

from __future__ import annotations

import re


NODE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:\[\[\s*\"?(.+?)\"?\s*\]\]|\[\s*\"?(.+?)\"?\s*\]|"
    r"\(\(\s*\"?(.+?)\"?\s*\)\)|\(\s*\"?(.+?)\"?\s*\)|"
    r"\{\s*\"?(.+?)\"?\s*\})"
)


def _terms(text: str) -> set[str]:
    compact = re.sub(r"<br\s*/?>|\s+|[\"'，。；：！？、（）()\[\]{}]", "", str(text).lower())
    latin = set(re.findall(r"[a-z0-9_\-]{2,}", compact))
    chinese_chars = "".join(re.findall(r"[\u4e00-\u9fff]", compact))
    chinese = {chinese_chars[index:index + 2] for index in range(len(chinese_chars) - 1)}
    return latin | chinese


def _contradicts(label: str, content: str, overlap: int) -> bool:
    if overlap <= 0:
        return False
    pattern = r"严禁|禁止|不得|不应|切勿|不要|never|must\s+not|do\s+not"
    return bool(re.search(pattern, label, re.I)) != bool(re.search(pattern, content, re.I))


def extract_nodes(diagram: str) -> list[dict]:
    nodes = []
    seen = set()
    for match in NODE_PATTERN.finditer(str(diagram or "")):
        node_id = match.group(1)
        if node_id in seen:
            continue
        label = next((value for value in match.groups()[1:] if value is not None), "")
        seen.add(node_id)
        nodes.append({"node_id": node_id, "label": label.strip().strip('"')})
    return nodes


def map_evidence(diagram: str, catalog: list[dict], top_k: int = 3) -> list[dict]:
    mappings = []
    for node in extract_nodes(diagram):
        node_terms = _terms(node["label"])
        scored = []
        for evidence in catalog:
            content = str(evidence.get("content") or evidence.get("snippet") or evidence.get("fault_description") or "")
            evidence_terms = _terms(content)
            overlap = len(node_terms & evidence_terms)
            if overlap:
                confidence = overlap / max(1, len(node_terms))
                trust = 1.0 if evidence.get("trust_level") == "authoritative" else float(evidence.get("credibility", 0.6))
                conflict = bool(evidence.get("conflict")) or _contradicts(node["label"], content, overlap)
                scored.append((confidence * (0.6 + 0.4 * trust), evidence, content, conflict))
        scored.sort(key=lambda item: item[0], reverse=True)
        links = [
            {
                "evidence_id": evidence.get("evidence_id") or evidence.get("source_id") or "",
                "source_type": evidence.get("source_type") or "external",
                "title": evidence.get("title") or evidence.get("fault_description") or "来源",
                "location": evidence.get("location") or evidence.get("url") or "",
                "confidence": round(score, 3),
                "quote": content[:240],
                "relation": "contradicts" if conflict else "supports",
            }
            for score, evidence, content, conflict in scored[:top_k]
        ]
        conflicts = [
            {"evidence_id": item["evidence_id"], "title": item["title"], "quote": item["quote"]}
            for item in links if item["relation"] == "contradicts"
        ]
        mappings.append({
            **node,
            "evidence": links,
            "confidence": round(max((item["confidence"] for item in links), default=0), 3),
            "needs_review": bool(conflicts) or not links or max((item["confidence"] for item in links), default=0) < 0.25,
            "conflicts": conflicts,
        })
    return mappings
