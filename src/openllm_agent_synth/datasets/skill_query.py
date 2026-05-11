# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Loong Ma

"""Built-in dataset implementation for skill routing query synthesis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import data_designer.config as dd
import pandas as pd
import yaml
from data_designer.config.seed_source_dataframe import DataFrameSeedSource

from openllm_agent_synth.config.models import ModelSettings, SkillQuerySpec

from .base import BuiltinDatasetBuilder, DatasetBuilderError

_FRONT_MATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+-]{1,}")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "agent",
    "always",
    "also",
    "and",
    "asks",
    "assist",
    "before",
    "build",
    "calls",
    "can",
    "chat",
    "codex",
    "create",
    "data",
    "default",
    "docs",
    "does",
    "file",
    "from",
    "help",
    "into",
    "latest",
    "need",
    "only",
    "questions",
    "requests",
    "return",
    "should",
    "skill",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "through",
    "tool",
    "tools",
    "use",
    "user",
    "using",
    "when",
    "with",
    "workflow",
}


@dataclass(frozen=True)
class SkillProfile:
    """Normalized representation of one skill directory."""

    skill_name: str
    skill_path: str
    skill_summary: str
    should_use_when: str
    should_not_use_when: str
    capability_tags: list[str]
    lexical_tokens: set[str]


class SkillQueryDatasetBuilder(BuiltinDatasetBuilder):
    """Build a synthetic dataset for skill routing and retrieval training."""

    dataset_type = "skill_query"
    spec_model = SkillQuerySpec

    def build(self, spec: SkillQuerySpec, model_settings: ModelSettings) -> dd.DataDesignerConfigBuilder:
        """Create the dataset config builder from dataset and model settings."""

        profiles = self._load_skill_profiles(spec)
        seed_frame = self._build_seed_frame(profiles, spec)

        model_config = dd.ModelConfig(
            alias=model_settings.alias,
            model=model_settings.name,
            provider=model_settings.provider,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                max_tokens=model_settings.max_tokens,
                extra_body={"reasoning_effort": model_settings.reasoning_effort},
            ),
            skip_health_check=model_settings.skip_health_check,
        )
        builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])
        builder.with_seed_dataset(DataFrameSeedSource(df=seed_frame))
        builder.add_column(
            dd.ExpressionColumnConfig(
                name="evidence_summary",
                expr=(
                    "{{ skill_summary }}"
                    "{% if should_use_when %} Use when: {{ should_use_when }}.{% endif %}"
                    "{% if should_not_use_when %} Avoid for: {{ should_not_use_when }}.{% endif %}"
                ),
            )
        )
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="query",
                model_alias=model_settings.alias,
                system_prompt=(
                    "You create realistic training queries for routing users to the right skill. "
                    "Write only the user query. Keep it self-contained, natural, and faithful to the skill scope."
                ),
                prompt=(
                    "Target skill: {{ target_skill }}\n"
                    "Query type: {{ query_type }}\n"
                    "Difficulty: {{ difficulty }}\n"
                    "Variant index: {{ query_index }}\n"
                    "Skill summary: {{ skill_summary }}\n"
                    "Use when: {{ should_use_when }}\n"
                    "Do not use when: {{ should_not_use_when }}\n"
                    "Capability tags: {{ capability_tags_json }}\n"
                    "Similar but wrong skills: {{ hard_negative_skills_json }}\n\n"
                    "Instructions:\n"
                    "- Produce one realistic user request that should route to the target skill.\n"
                    "- The query must not mention internal file paths.\n"
                    "- For `direct`, make the user intent explicit.\n"
                    "- For `goal_oriented`, describe the outcome without naming the skill.\n"
                    "- For `discovery`, ask which skill/workflow can help.\n"
                    "- For `ambiguous`, keep it plausible but less direct while still routing correctly.\n"
                    "- Keep the query concise, specific, and distinct from obvious paraphrases.\n"
                    "Return only the query text."
                ),
                with_trace=dd.TraceType.NONE,
            )
        )
        builder.add_column(
            dd.LLMTextColumnConfig(
                name="routing_rationale",
                model_alias=model_settings.alias,
                system_prompt=(
                    "You explain routing labels for a skill-query training dataset. Be concise and factual."
                ),
                prompt=(
                    "Given this query:\n{{ query }}\n\n"
                    "Target skill: {{ target_skill }}\n"
                    "Skill summary: {{ skill_summary }}\n"
                    "Use when: {{ should_use_when }}\n"
                    "Do not use when: {{ should_not_use_when }}\n"
                    "Similar but wrong skills: {{ hard_negative_skills_json }}\n\n"
                    "Write 1-2 sentences explaining why the target skill is the best routing choice."
                ),
                with_trace=dd.TraceType.NONE,
            )
        )
        return builder

    def _load_skill_profiles(self, spec: SkillQuerySpec) -> list[SkillProfile]:
        """Scan skill sources and return normalized profiles."""

        skill_dirs = self._resolve_skill_dirs(spec)
        if not skill_dirs:
            raise DatasetBuilderError("No skill directories were found from the configured `skill_dirs`/`skill_roots`.")

        profiles = [self._parse_skill_profile(skill_dir, spec.max_skill_content_chars) for skill_dir in skill_dirs]
        profile_names = [profile.skill_name for profile in profiles]
        if len(profile_names) != len(set(profile_names)):
            duplicates = sorted({name for name in profile_names if profile_names.count(name) > 1})
            raise DatasetBuilderError(
                "Duplicate skill names were found while building skill_query: " + ", ".join(duplicates)
            )
        if len(profiles) == 1 and spec.include_negative_samples and spec.negatives_per_query > 0:
            raise DatasetBuilderError("At least two skills are required when `include_negative_samples` is enabled.")
        return profiles

    def _resolve_skill_dirs(self, spec: SkillQuerySpec) -> list[Path]:
        """Expand explicit directories and root scans into a de-duplicated path list."""

        resolved: dict[Path, None] = {}
        for raw_dir in spec.skill_dirs:
            skill_dir = Path(raw_dir).expanduser().resolve()
            if not skill_dir.exists():
                raise DatasetBuilderError(f"Configured skill directory does not exist: {skill_dir}")
            if not skill_dir.is_dir():
                raise DatasetBuilderError(f"Configured skill directory is not a directory: {skill_dir}")
            if not (skill_dir / "SKILL.md").exists():
                raise DatasetBuilderError(f"Configured skill directory is missing SKILL.md: {skill_dir}")
            resolved[skill_dir] = None

        for raw_root in spec.skill_roots:
            skill_root = Path(raw_root).expanduser().resolve()
            if not skill_root.exists():
                raise DatasetBuilderError(f"Configured skill root does not exist: {skill_root}")
            if not skill_root.is_dir():
                raise DatasetBuilderError(f"Configured skill root is not a directory: {skill_root}")
            for skill_file in sorted(skill_root.rglob("SKILL.md")):
                resolved[skill_file.parent.resolve()] = None

        return list(resolved)

    def _parse_skill_profile(self, skill_dir: Path, max_chars: int) -> SkillProfile:
        """Read one SKILL.md file and normalize its routing signals."""

        skill_path = skill_dir / "SKILL.md"
        raw_text = skill_path.read_text(encoding="utf-8")
        front_matter, body = self._split_front_matter(raw_text)

        skill_name = str(front_matter.get("name") or self._extract_heading(body) or skill_dir.name).strip()
        description = str(front_matter.get("description") or "").strip()
        fallback_summary = self._first_paragraph(body)
        skill_summary = self._collapse_text(description or fallback_summary or f"Skill located at {skill_dir.name}")
        should_use_when = self._extract_guidance(
            body,
            fallback_prefixes=("use when", "use this skill", "when the user", "use for"),
        )
        should_not_use_when = self._extract_guidance(
            body,
            fallback_prefixes=("do not use", "avoid", "instead", "only use"),
        )
        summary_context = self._collapse_text(body)[:max_chars]
        lexical_tokens = self._tokenize(" ".join([skill_name, skill_summary, summary_context]))
        capability_tags = self._derive_capability_tags(skill_name, skill_summary, lexical_tokens)

        return SkillProfile(
            skill_name=skill_name,
            skill_path=str(skill_path),
            skill_summary=skill_summary,
            should_use_when=should_use_when,
            should_not_use_when=should_not_use_when,
            capability_tags=capability_tags,
            lexical_tokens=lexical_tokens,
        )

    def _build_seed_frame(self, profiles: list[SkillProfile], spec: SkillQuerySpec) -> pd.DataFrame:
        """Create seed rows for all skill/query variants before LLM generation."""

        negatives_by_skill = self._build_negative_map(
            profiles, spec.negatives_per_query if spec.include_negative_samples else 0
        )
        rows: list[dict[str, object]] = []
        difficulty_levels = spec.difficulty_levels
        query_types = spec.query_types

        for profile in profiles:
            for index in range(spec.queries_per_skill):
                query_type = query_types[index % len(query_types)]
                difficulty = difficulty_levels[(index // len(query_types)) % len(difficulty_levels)]
                normalized_name = profile.skill_name.lower().replace(" ", "-").replace(":", "-")
                rows.append(
                    {
                        "sample_id": f"sq-{normalized_name}-{index + 1:03d}",
                        "target_skill": profile.skill_name,
                        "skill_path": profile.skill_path,
                        "skill_summary": profile.skill_summary,
                        "should_use_when": profile.should_use_when,
                        "should_not_use_when": profile.should_not_use_when,
                        "capability_tags_json": json.dumps(profile.capability_tags, ensure_ascii=True),
                        "query_type": query_type,
                        "difficulty": difficulty,
                        "query_index": index + 1,
                        "hard_negative_skills_json": json.dumps(
                            negatives_by_skill.get(profile.skill_name, []), ensure_ascii=True
                        ),
                    }
                )

        return pd.DataFrame(rows)

    def _build_negative_map(self, profiles: list[SkillProfile], negatives_per_query: int) -> dict[str, list[str]]:
        """Select a few lexically similar skills as hard negatives."""

        if negatives_per_query <= 0:
            return {profile.skill_name: [] for profile in profiles}

        negative_map: dict[str, list[str]] = {}
        for profile in profiles:
            scored_candidates: list[tuple[int, str]] = []
            for other in profiles:
                if other.skill_name == profile.skill_name:
                    continue
                overlap = len(profile.lexical_tokens & other.lexical_tokens)
                scored_candidates.append((overlap, other.skill_name))

            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [name for _, name in scored_candidates[:negatives_per_query]]
            negative_map[profile.skill_name] = selected
        return negative_map

    def _split_front_matter(self, raw_text: str) -> tuple[dict[str, object], str]:
        """Split YAML front matter from the markdown body when present."""

        match = _FRONT_MATTER_PATTERN.match(raw_text)
        if not match:
            return {}, raw_text

        try:
            front_matter = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise DatasetBuilderError("Invalid YAML front matter in SKILL.md") from exc
        if not isinstance(front_matter, dict):
            raise DatasetBuilderError("SKILL.md front matter must be a YAML mapping.")
        return front_matter, raw_text[match.end() :]

    def _extract_heading(self, body: str) -> str:
        """Extract the first H1 title from the markdown body."""

        match = _HEADING_PATTERN.search(body)
        return match.group(1).strip() if match else ""

    def _first_paragraph(self, body: str) -> str:
        """Return the first non-heading paragraph from the markdown body."""

        for paragraph in body.split("\n\n"):
            text = self._collapse_text(paragraph)
            if text and not text.startswith("#"):
                return text
        return ""

    def _extract_guidance(self, body: str, *, fallback_prefixes: tuple[str, ...]) -> str:
        """Pull a few guidance lines from the skill body for routing prompts."""

        collected: list[str] = []
        for raw_line in body.splitlines():
            line = self._collapse_text(raw_line.lstrip("-* ").strip())
            lowered = line.lower()
            if any(prefix in lowered for prefix in fallback_prefixes):
                collected.append(line.rstrip("."))
            if len(collected) == 3:
                break
        return " ".join(collected)

    def _collapse_text(self, text: str) -> str:
        """Normalize whitespace inside free-form markdown content."""

        return re.sub(r"\s+", " ", text).strip()

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize free-form skill content for simple lexical matching."""

        tokens = {token.lower() for token in _TOKEN_PATTERN.findall(text.lower())}
        return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}

    def _derive_capability_tags(self, skill_name: str, skill_summary: str, tokens: set[str]) -> list[str]:
        """Create a small stable set of capability tags from local metadata."""

        preferred_tokens = []
        for token in _TOKEN_PATTERN.findall(skill_name.lower()):
            preferred_tokens.extend(token.split("-"))
        for token in _TOKEN_PATTERN.findall(skill_summary.lower()):
            preferred_tokens.extend(token.split("-"))

        tags: list[str] = []
        for token in preferred_tokens:
            cleaned = token.strip("_")
            if cleaned and cleaned not in _STOPWORDS and len(cleaned) > 2 and cleaned in tokens and cleaned not in tags:
                tags.append(cleaned)
            if len(tags) == 8:
                break

        if not tags:
            tags = sorted(tokens)[:8]
        return tags
