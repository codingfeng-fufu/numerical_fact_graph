from __future__ import annotations

import re


def strip_negative_distractors(query: str) -> str:
    cleaned = re.sub(
        r"(?:;|,)?\s*(?:do\s+not|don't|ignore|not)\s+[^.;,]*(?:growth|margin|total|count|attribute|field|column)[^.;,]*",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or query
