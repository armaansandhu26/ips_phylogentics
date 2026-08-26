"""Unified CLI entry: python -m final <command> ..."""

from __future__ import annotations

import sys
import json


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m final <command> [args...]\n\n"
            "Commands:\n"
            "  run_suite        Run all four methods for a suite\n"
            "  pipeline         Train + sample + plots for one method\n"
            "  aggregate        Build comparison tables from completed runs\n"
            "  preflight        Run m(x) + topology verification\n"
            "  verify-mx        Verify 5-taxa m(x) enumeration\n"
            "  verify-topology  Verify topology hash + Newick\n"
            "  hypergrid-pipeline  Train + eval GRPO or count IPS on Hyper-Grid\n",
            flush=True,
        )
        raise SystemExit(2)

    command = sys.argv[1]
    argv = sys.argv[2:]

    if command == "run_suite":
        from final.run_suite import main as run_suite_main

        run_suite_main(argv)
    elif command == "pipeline":
        from final.pipeline import main as pipeline_main

        pipeline_main(argv)
    elif command == "aggregate":
        from final.aggregate import main as aggregate_main

        aggregate_main(argv)
    elif command == "preflight":
        from final.configs import load_suite
        from final.preflight import run_preflight

        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--suite", required=True)
        args = p.parse_args(argv)
        run_preflight(load_suite(args.suite))
    elif command == "verify-mx":
        from final.verify.verify_mx import main as verify_mx_main

        verify_mx_main(argv)
    elif command == "verify-topology":
        from final.verify.verify_topology_hash import main as verify_topo_main

        verify_topo_main(argv)
    elif command == "list":
        from final.configs import list_suites, load_suite

        for path in list_suites():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("env") == "hypergrid":
                print(
                    f"{raw['id']:20s}  env=hypergrid  "
                    f"methods={','.join(raw.get('methods', []))}  "
                    f"epochs={raw['training']['epochs']}"
                )
                continue
            suite = load_suite(path)
            print(
                f"{suite.id:20s}  taxa={suite.taxa:2d}  "
                f"shift={suite.log_score_shift:g}  epochs={suite.training.epochs}"
            )
    elif command == "hypergrid-pipeline":
        from final.toy.pipeline import main as hypergrid_pipeline_main

        hypergrid_pipeline_main(argv)
    else:
        print(f"unknown command: {command}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
