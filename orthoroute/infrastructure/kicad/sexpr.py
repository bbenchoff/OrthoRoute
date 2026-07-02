"""Minimal s-expression tokenizer/parser for KiCad board files.

KiCad `.kicad_pcb` files are s-expressions: nested parenthesized lists of
atoms and double-quoted strings (with backslash escapes). The legacy regex
extractors in ``file_parser.py`` broke on every format revision; this module
replaces them with a single balanced-paren parser that works for every
dialect from KiCad 5 through KiCad 10.

Two independent entry points:

- :func:`parse` / :func:`parse_file` — full structural parse into nested
  Python lists (used by the KiCad-10-capable file parser).
- :func:`strip_top_level_nodes` — byte-exact removal of selected top-level
  nodes from the raw text without reserialization (used to produce stripped
  test fixtures whose untouched content is byte-identical to the original).

This module must stay Python 3.9 compatible: it is imported by plugin
runtime code and macOS KiCad bundles Python 3.9.
"""

from typing import List, Optional, Tuple, Union

# A parsed node is a list whose first element is usually the node name atom;
# leaves are strings (atoms and unescaped quoted strings are not
# distinguished — KiCad consumers never need the distinction).
SExpr = Union[str, List["SExpr"]]

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "n": "\n",
    "t": "\t",
    "r": "\r",
}


class SExprError(ValueError):
    """Raised on malformed s-expression input (unbalanced parens, EOF in string)."""


def parse(text: str) -> List[SExpr]:
    """Parse s-expression text into a list of top-level nodes.

    Args:
        text: Full file contents.

    Returns:
        List of top-level expressions (a .kicad_pcb has exactly one:
        the ``kicad_pcb`` node).

    Raises:
        SExprError: On unbalanced parentheses or an unterminated string.
    """
    top: List[SExpr] = []
    stack: List[List[SExpr]] = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c == "(":
            node: List[SExpr] = []
            if stack:
                stack[-1].append(node)
            else:
                top.append(node)
            stack.append(node)
            i += 1
        elif c == ")":
            if not stack:
                raise SExprError(f"Unbalanced ')' at offset {i}")
            stack.pop()
            i += 1
        elif c == '"':
            value, i = _read_string(text, i)
            if stack:
                stack[-1].append(value)
            else:
                top.append(value)
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()"':
                j += 1
            atom = text[i:j]
            if stack:
                stack[-1].append(atom)
            else:
                top.append(atom)
            i = j

    if stack:
        raise SExprError(f"Unbalanced '(': {len(stack)} unclosed at EOF")
    return top


def parse_file(path: str) -> List[SExpr]:
    """Parse a file; see :func:`parse`."""
    with open(path, "r", encoding="utf-8") as f:
        return parse(f.read())


def _read_string(text: str, i: int) -> Tuple[str, int]:
    """Read a quoted string starting at ``text[i] == '"'``.

    Returns (unescaped value, index just past the closing quote).
    """
    n = len(text)
    i += 1  # opening quote
    out: List[str] = []
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            nxt = text[i + 1]
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
        elif c == '"':
            return "".join(out), i + 1
        else:
            out.append(c)
            i += 1
    raise SExprError("Unterminated string at EOF")


# ---------------------------------------------------------------------------
# Structural helpers for extractors
# ---------------------------------------------------------------------------

def node_name(node: SExpr) -> Optional[str]:
    """Return the name atom of a node, or None for leaves/empty nodes."""
    if isinstance(node, list) and node and isinstance(node[0], str):
        return node[0]
    return None


def children(node: SExpr, name: str) -> List[List[SExpr]]:
    """All direct child nodes of ``node`` named ``name``."""
    if not isinstance(node, list):
        return []
    return [c for c in node[1:] if isinstance(c, list) and c and c[0] == name]


def child(node: SExpr, name: str) -> Optional[List[SExpr]]:
    """First direct child node named ``name``, or None."""
    found = children(node, name)
    return found[0] if found else None


def atoms(node: SExpr) -> List[str]:
    """Leaf values (atoms/strings) of a node, excluding the name atom."""
    if not isinstance(node, list):
        return []
    return [c for c in node[1:] if isinstance(c, str)]


def first_atom(node: Optional[SExpr]) -> Optional[str]:
    """First leaf value of a node, or None."""
    if node is None:
        return None
    vals = atoms(node)
    return vals[0] if vals else None


# ---------------------------------------------------------------------------
# Byte-exact top-level node stripping (fixture generation)
# ---------------------------------------------------------------------------

def find_top_level_spans(text: str, names: Tuple[str, ...]) -> List[Tuple[int, int]]:
    """Find byte spans of depth-1 nodes named in ``names``.

    Depth 1 means direct children of the single root ``(kicad_pcb ...)``
    node — where KiCad keeps segments, vias, zones, and footprints.

    Each span covers the node's opening ``(`` through its closing ``)``,
    widened to include the preceding same-line indentation and the trailing
    newline, so removal deletes whole lines without leaving blanks.
    """
    spans: List[Tuple[int, int]] = []
    depth = 0
    i = 0
    n = len(text)

    while i < n:
        c = text[i]
        if c == '"':
            _, i = _read_string(text, i)
        elif c == "(":
            if depth == 1:
                j = i + 1
                while j < n and text[j] not in ' \t\r\n()"':
                    j += 1
                if text[i + 1:j] in names:
                    end = _scan_node_end(text, i)
                    start = i
                    while start > 0 and text[start - 1] in " \t":
                        start -= 1
                    if end < n and text[end] == "\r":
                        end += 1
                    if end < n and text[end] == "\n":
                        end += 1
                    spans.append((start, end))
                    i = end
                    continue
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                raise SExprError(f"Unbalanced ')' at offset {i}")
            i += 1
        else:
            i += 1

    if depth != 0:
        raise SExprError(f"Unbalanced '(': depth {depth} at EOF")
    return spans


def _scan_node_end(text: str, start: int) -> int:
    """Given ``text[start] == '('``, return index just past the matching ')'."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == '"':
            _, i = _read_string(text, i)
        elif c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            i += 1
            if depth == 0:
                return i
        else:
            i += 1
    raise SExprError("Unbalanced '(' while scanning node")


def strip_top_level_nodes(text: str, names: Tuple[str, ...]) -> Tuple[str, int]:
    """Remove all depth-1 nodes named in ``names`` from raw file text.

    Untouched content is preserved byte-for-byte (no reserialization), so
    stripping is deterministic and re-runnable: stripping an already
    stripped file is a no-op.

    Returns:
        (stripped text, number of nodes removed)
    """
    spans = find_top_level_spans(text, names)
    if not spans:
        return text, 0
    out: List[str] = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        prev = end
    out.append(text[prev:])
    return "".join(out), len(spans)
