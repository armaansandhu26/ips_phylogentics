#!/usr/bin/env python3
"""Create a reduced DS1 pickle by keeping the first N species in file order."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "DS1.pickle",
    )
    parser.add_argument(
        "--num-species",
        type=int,
        required=True,
        help="Number of leading species to keep from DS1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: DS1_reduced_{num_species}taxa.pickle next to --input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_species <= 0:
        raise ValueError("--num-species must be positive")

    with args.input.open("rb") as handle:
        species_to_sequence = pickle.load(handle)
    if not isinstance(species_to_sequence, dict):
        raise TypeError(f"expected dict in {args.input}, got {type(species_to_sequence)}")

    species_names = list(species_to_sequence.keys())
    if args.num_species > len(species_names):
        raise ValueError(
            f"requested {args.num_species} species but {args.input} has {len(species_names)}"
        )

    selected_names = species_names[: args.num_species]
    reduced = {name: species_to_sequence[name] for name in selected_names}

    output = args.output or args.input.with_name(
        f"DS1_reduced_{args.num_species}taxa.pickle"
    )
    with output.open("wb") as handle:
        pickle.dump(reduced, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequence_lengths = {len(seq) for seq in reduced.values()}
    print(f"wrote {output}")
    print(f"species: {len(reduced)}")
    print(f"sequence lengths: {sorted(sequence_lengths)}")
    print("species names:")
    for name in selected_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
