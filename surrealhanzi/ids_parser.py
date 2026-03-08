"""Parse IDS (Ideographic Description Sequence) strings into trees."""

from dataclasses import dataclass, field
from typing import Optional

# IDS operators and their arities
IDS_OPERATORS: dict[str, int] = {
    "\u2FF0": 2,  # ⿰ Left to Right
    "\u2FF1": 2,  # ⿱ Above to Below
    "\u2FF2": 3,  # ⿲ Left to Middle and Right
    "\u2FF3": 3,  # ⿳ Above to Middle and Below
    "\u2FF4": 2,  # ⿴ Full Surround
    "\u2FF5": 2,  # ⿵ Surround from Above
    "\u2FF6": 2,  # ⿶ Surround from Below
    "\u2FF7": 2,  # ⿷ Surround from Left
    "\u2FF8": 2,  # ⿸ Surround from Upper Left
    "\u2FF9": 2,  # ⿹ Surround from Upper Right
    "\u2FFA": 2,  # ⿺ Surround from Lower Left
    "\u2FFB": 2,  # ⿻ Overlaid
    # Unicode 15.1+ additions
    "\u2FFC": 2,  # ⿼ Surround from Right
    "\u2FFD": 2,  # ⿽ Surround from Below-Right (horizontal reflection)
    "\u2FFE": 1,  # ⿾ Mirror (unary, horizontal reflection)
    "\u2FFF": 1,  # ⿿ Rotation (unary, 180° rotation)
    "\u31EF": 2,  # ㇯ Subtraction (binary, remove second from first)
}


def is_ids_operator(ch: str) -> bool:
    return ch in IDS_OPERATORS


@dataclass
class IDSNode:
    """A node in an IDS parse tree.

    Either a leaf (character is set, children is empty) or an operator node
    (operator is set, children contains the operands).
    """
    operator: Optional[str] = None
    character: Optional[str] = None
    children: list["IDSNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.operator is None

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf({self.character})"
        kids = ", ".join(repr(c) for c in self.children)
        return f"Op({self.operator}, [{kids}])"

    def to_ids(self) -> str:
        """Reconstruct the IDS string from this tree."""
        if self.is_leaf:
            return self.character or ""
        parts = [self.operator or ""]
        for child in self.children:
            parts.append(child.to_ids())
        return "".join(parts)


class IDSParseError(Exception):
    pass


def parse_ids(ids_string: str) -> IDSNode:
    """Parse an IDS string into a tree.

    IDS uses prefix notation: operator followed by N operands,
    where operands can themselves be IDS expressions.

    Examples:
        "⿰女子" -> Op(⿰, [Leaf(女), Leaf(子)])
        "⿱⿰日月火" -> Op(⿱, [Op(⿰, [Leaf(日), Leaf(月)]), Leaf(火)])
    """
    chars = list(ids_string)
    pos = [0]  # mutable index

    def _parse() -> IDSNode:
        if pos[0] >= len(chars):
            raise IDSParseError(f"Unexpected end of IDS string: {ids_string!r}")

        ch = chars[pos[0]]
        pos[0] += 1

        if is_ids_operator(ch):
            arity = IDS_OPERATORS[ch]
            children = []
            for _ in range(arity):
                children.append(_parse())
            return IDSNode(operator=ch, children=children)
        else:
            return IDSNode(character=ch)

    tree = _parse()

    if pos[0] < len(chars):
        remaining = "".join(chars[pos[0]:])
        raise IDSParseError(
            f"Trailing characters after complete IDS expression: {remaining!r}"
        )

    return tree


def collect_leaves(node: IDSNode) -> list[str]:
    """Collect all leaf characters from an IDS tree."""
    if node.is_leaf:
        return [node.character] if node.character else []
    result = []
    for child in node.children:
        result.extend(collect_leaves(child))
    return result
