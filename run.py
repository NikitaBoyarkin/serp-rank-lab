"""One entry point for the SERP Rank Lab pipeline (Phase 3).

    python run.py --stage {ingest,sessions,serp,rank,eval,experiment,all}

Stage -> module
---------------
``ingest``      ``src.data_ingest``   -- real feeds -> ``data/processed/*.parquet``
``sessions``    ``src.sessions``      -- events -> ``data/processed/sessions.parquet``
``serp``        ``src.serp_sim``      -- ``serp_offers.parquet``
``rank``        ``src.ranking_eval``  -- ``reports/ranking_eval.json``
``eval``        alias of ``rank``     -- the requested CLI surface names both; they
                                        run the same ranking-evaluation module
``experiment``  ``src.experiment``    -- ``reports/experiment.json`` + ``reports/peeking.png``
``sql``         ``src.sql_stage``     -- ``sql/03``+``04`` -> ``data/processed/<view>.parquet``
``business``    ``src.business``      -- real islands + clickstream -> ``reports/business.json``
                                        + ``reports/figures/*.png``
``all``         every stage, in dependency order

``src/sessions.py`` exposes ``build_sessions`` but no ``main()``; the ``sessions``
stage wires it here (events -> ``sessions.parquet``) rather than editing the
verified module. ``--stage all`` runs end-to-end from a clone whose only data is
``data/raw/`` (the raw real feeds themselves live outside the repo and are read in
place by ``src/data_ingest``).
"""

from __future__ import annotations

import argparse
import importlib

import pandas as pd

# Stage -> importable module exposing ``main()``. ``eval`` reuses ``rank``'s module.
_STAGE_MODULE = {
    "ingest": "src.data_ingest",
    "serp": "src.serp_sim",
    "rank": "src.ranking_eval",
    "eval": "src.ranking_eval",
    "experiment": "src.experiment",
    "sql": "src.sql_stage",
    "business": "src.business",
}

# Dependency order for ``all`` (``eval`` is an alias, so it is not repeated).
_ALL_ORDER = ["ingest", "sessions", "serp", "sql", "rank", "experiment", "business"]


def _run_sessions() -> int:
    """Sessionise ``events.parquet`` and write ``data/processed/sessions.parquet``.

    Mirrors what a ``src.sessions`` CLI would do; the module keeps its library
    surface unchanged.
    """
    from src.data_ingest import PROCESSED_DIR
    from src.sessions import build_sessions

    events_path = PROCESSED_DIR / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}. Run `run.py --stage ingest` first.")

    print("=== sessions ===")
    print(f"source: {events_path}")
    events = pd.read_parquet(events_path)
    sessions = build_sessions(events)
    target = PROCESSED_DIR / "sessions.parquet"
    sessions.to_parquet(target, index=False)
    print(f"rows:   {len(sessions):,}")
    print(f"written: {target}")
    return 0


def run_stage(stage: str) -> int:
    """Dispatch one stage to its module's ``main()`` (or the sessions writer)."""
    if stage == "sessions":
        return _run_sessions()
    module = importlib.import_module(_STAGE_MODULE[stage])
    print(f"\n### stage: {stage} -> {_STAGE_MODULE[stage]}")
    # Pass an explicit argv so the stage's own argparse never sees run.py's flags.
    return int(module.main([]) or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="SERP Rank Lab pipeline entry point",
    )
    parser.add_argument(
        "--stage",
        choices=[*_STAGE_MODULE, "sessions", "all"],
        default="all",
        help="pipeline stage to run (default: all)",
    )
    args = parser.parse_args(argv)

    stages = _ALL_ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        rc = run_stage(stage)
        if rc != 0:
            return rc
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
