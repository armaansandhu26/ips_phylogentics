"""Paper-compatible molecular discovery metrics for non-enumerable tasks."""

from __future__ import annotations

from statistics import fmean


def _threshold_key(value: float) -> str:
    return str(value).rstrip("0").rstrip(".").replace("-", "neg_").replace(".", "p")


def molecular_discovery_metrics(
    rows: list[dict],
    *,
    mode_threshold: float,
    similarity_threshold: float = 0.5,
    max_modes: int = 5000,
    top_k: int = 500,
    scaffold_thresholds: tuple[float, ...] = (),
) -> tuple[dict[str, float | int | bool], list[dict]]:
    """Compute the RGFN paper's leader modes, scaffolds, and top-mode properties."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:  # pragma: no cover - depends on RDKit packaging
        sascorer = None

    best_by_smiles: dict[str, dict] = {}
    for row in rows:
        smiles = row.get("smiles")
        if smiles is None:
            continue
        previous = best_by_smiles.get(smiles)
        if previous is None or float(row["proxy"]) > float(previous["proxy"]):
            best_by_smiles[str(smiles)] = row

    parsed: list[tuple[str, dict, object]] = []
    n_invalid = 0
    for smiles, row in best_by_smiles.items():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            n_invalid += 1
            continue
        parsed.append((smiles, row, molecule))
    parsed.sort(key=lambda item: float(item[1]["proxy"]), reverse=True)

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    candidates = [item for item in parsed if float(item[1]["proxy"]) > mode_threshold]
    leader_fingerprints = []
    modes: list[dict] = []
    for smiles, row, molecule in candidates:
        fingerprint = fingerprint_generator.GetFingerprint(molecule)
        if leader_fingerprints:
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprint, leader_fingerprints
            )
            if max(similarities) > similarity_threshold:
                continue
        scaffold = MurckoScaffoldSmiles(smiles=smiles)
        mode = {
            "smiles": smiles,
            "proxy": float(row["proxy"]),
            "reward": float(row["reward"]),
            "qed": float(QED.qed(molecule)),
            "molecular_weight": float(Descriptors.MolWt(molecule)),
            "scaffold": scaffold,
        }
        if sascorer is not None:
            mode["sa_score"] = float(sascorer.calculateScore(molecule))
        modes.append(mode)
        leader_fingerprints.append(fingerprint)
        if len(modes) >= max_modes:
            break

    top_modes = modes[:top_k]
    metrics: dict[str, float | int | bool] = {
        "mode_proxy_threshold": mode_threshold,
        "mode_similarity_threshold": similarity_threshold,
        "n_mode_candidates": len(candidates),
        "n_modes": len(modes),
        "modes_capped": len(modes) >= max_modes,
        "n_invalid_unique_smiles": n_invalid,
        "top_mode_k_requested": top_k,
        "top_mode_k_available": len(top_modes),
    }
    if top_modes:
        for key in ("proxy", "reward", "qed", "molecular_weight", "sa_score"):
            values = [float(mode[key]) for mode in top_modes if key in mode]
            if values:
                metrics[f"top_modes_mean_{key}"] = fmean(values)
        metrics["top_modes_unique_scaffolds"] = len(
            {mode["scaffold"] for mode in top_modes if mode["scaffold"]}
        )

    for threshold in scaffold_thresholds:
        scaffolds = {
            MurckoScaffoldSmiles(smiles=smiles)
            for smiles, row, _ in parsed
            if float(row["proxy"]) > threshold
        }
        scaffolds.discard("")
        metrics[f"n_scaffolds_proxy_gt_{_threshold_key(threshold)}"] = len(scaffolds)

    return metrics, modes
