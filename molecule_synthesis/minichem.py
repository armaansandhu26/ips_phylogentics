"""Deterministic reduced chemistry vocabulary for exact CPU experiments."""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("RGFN_MINIMAL_PROXIES", "1")

import gin
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles

from rgfn.gfns.reaction_gfn.api.data_structures import AnchoredReaction, Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import Reaction


@gin.configurable()
class MiniReactionDataFactory:
    """Load a small balanced subset of the released RGFN chemical language.

    The selection is stable with respect to spreadsheet row order and avoids
    the non-deterministic ``set`` conversion in the upstream data factory.
    """

    def __init__(
        self,
        reaction_path: str | Path,
        reaction_families: Sequence[str] = ("Amide synthesis",),
        fragment_groups: Sequence[str] = ("Acids", "Amines"),
        fragments_per_group: int = 8,
        docking: bool = False,
    ):
        if fragments_per_group <= 0:
            raise ValueError("fragments_per_group must be positive")
        reaction_path = Path(reaction_path)
        reaction_sheet = "Reactions_Docking" if docking else "Reactions_NoDocking"
        fragment_sheet = "Fragments_Docking" if docking else "Fragments_NoDocking"

        reaction_frame = pd.read_excel(reaction_path, sheet_name=reaction_sheet)
        reaction_frame["Family"] = reaction_frame["Family"].ffill()
        selected_reactions = reaction_frame[
            reaction_frame["Family"].isin(tuple(reaction_families))
        ]["Reaction"].tolist()
        selected_reactions = [value for value in selected_reactions if isinstance(value, str)]
        if not selected_reactions:
            raise ValueError(f"No reactions found for families {tuple(reaction_families)!r}")

        self.reactions = [Reaction(smarts, idx) for idx, smarts in enumerate(selected_reactions)]
        self.disconnections = [reaction.reversed() for reaction in self.reactions]
        self.anchored_reactions: list[AnchoredReaction] = []
        self.reaction_anchor_map: Dict[Tuple[Reaction, int], AnchoredReaction] = {}
        for reaction in self.reactions:
            for anchor_idx in range(len(reaction.left_side_patterns)):
                anchored = AnchoredReaction(
                    reaction=reaction.reaction,
                    idx=len(self.anchored_reactions),
                    anchor_pattern_idx=anchor_idx,
                )
                self.reaction_anchor_map[(reaction, anchor_idx)] = anchored
                self.anchored_reactions.append(anchored)
        self.anchored_disconnections = [reaction.reversed() for reaction in self.anchored_reactions]

        fragment_frame = pd.read_excel(reaction_path, sheet_name=fragment_sheet)
        fragment_frame["Group"] = fragment_frame["Group"].ffill()
        canonical_smiles: list[str] = []
        seen: set[str] = set()
        for group in fragment_groups:
            group_values = fragment_frame[fragment_frame["Group"] == group]["Fragment"]
            selected_in_group = 0
            for value in group_values:
                molecule = MolFromSmiles(value) if isinstance(value, str) else None
                if molecule is None:
                    continue
                smiles = MolToSmiles(molecule)
                if smiles in seen:
                    continue
                seen.add(smiles)
                canonical_smiles.append(smiles)
                selected_in_group += 1
                if selected_in_group == fragments_per_group:
                    break
            if selected_in_group < fragments_per_group:
                raise ValueError(
                    f"Fragment group {group!r} has only {selected_in_group} usable unique entries"
                )

        self.fragments = [Molecule(smiles, idx=idx) for idx, smiles in enumerate(canonical_smiles)]
        print(
            "Using MiniChem with "
            f"{len(self.fragments)} fragments, {len(self.reactions)} reactions, "
            f"and {len(self.anchored_reactions)} anchored reactions"
        )

    def get_reactions(self) -> List[Reaction]:
        return copy(self.reactions)

    def get_disconnections(self) -> List[Reaction]:
        return copy(self.disconnections)

    def get_anchored_reactions(self) -> List[AnchoredReaction]:
        return copy(self.anchored_reactions)

    def get_reaction_anchor_map(self) -> Dict[Tuple[Reaction, int], AnchoredReaction]:
        return copy(self.reaction_anchor_map)

    def get_anchored_disconnections(self) -> List[AnchoredReaction]:
        return copy(self.anchored_disconnections)

    def get_fragments(self) -> List[Molecule]:
        return copy(self.fragments)
