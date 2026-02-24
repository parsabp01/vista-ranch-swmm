from __future__ import annotations

import re
from typing import Literal

IdKind = Literal['junction', 'inlet']


def norm_numeric_id(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {'nan', 'none'}:
        return None
    m = re.search(r'\d+(?:\.\d+)?', s)
    if not m:
        return None
    n = float(m.group(0))
    return str(int(n)) if n.is_integer() else str(n).rstrip('0').rstrip('.')


def typed_canonical_id(value: object, kind: IdKind) -> str | None:
    base = norm_numeric_id(value)
    if not base:
        return None
    token = base.replace('.', '_')
    prefix = 'J_' if kind == 'junction' else 'IN_'
    return f'{prefix}{token}'


def swmm_junction_id(value: object) -> str | None:
    base = norm_numeric_id(value)
    if not base:
        return None
    n = float(base)
    return f"J{int(n):03d}" if n.is_integer() else f"J{base.replace('.', '_')}"
