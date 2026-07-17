from __future__ import annotations

import re
import unicodedata


_PART_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"^[\u3400-\u9fff]+$")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """Create mixed-language tokens, including Chinese 1/2/3-grams."""

    tokens: list[str] = []
    for part in _PART_PATTERN.findall(normalize_text(text)):
        if not _CJK_PATTERN.fullmatch(part):
            tokens.append(f"w:{part}")
            continue
        for char in part:
            tokens.append(f"c1:{char}")
        for width in (2, 3):
            for index in range(max(0, len(part) - width + 1)):
                tokens.append(f"c{width}:{part[index:index + width]}")
        if len(part) <= 8:
            tokens.append(f"cf:{part}")
    return tokens


def display_token(token: str) -> str:
    return token.split(":", 1)[-1]
