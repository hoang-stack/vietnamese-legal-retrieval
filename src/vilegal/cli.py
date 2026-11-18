"""Command line: ``vilegal {experiments,search,serve,report}``."""

from __future__ import annotations

import argparse
import logging


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="vilegal", description=__doc__)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("experiments", help="evaluate BM25 (CPU) and the Kaggle runs")
    ex.add_argument("stages", nargs="*", default=["bm25", "neural"])
    ex.add_argument("--results-dir", default="results")
    ex.add_argument("--kaggle-dir", default="kaggle/output/main")
    se = sub.add_parser("search", help="search the corpus from the terminal")
    se.add_argument("query", nargs="+")
    se.add_argument("-k", type=int, default=5)
    sv = sub.add_parser("serve", help="REST API + search page")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    rp = sub.add_parser("report", help="figures + reports/results.md")
    rp.add_argument("--results-dir", default="results")
    rp.add_argument("--report-dir", default="reports")
    rp.add_argument("--kaggle-dir", default="kaggle/output/main")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "experiments":
        from vilegal.experiments import run

        run(args.stages, args.data_dir, args.results_dir, args.kaggle_dir)
    elif args.cmd == "search":
        from vilegal import data
        from vilegal.search import SearchEngine

        ds = data.load(args.data_dir)
        for h in SearchEngine(ds.doc_ids, ds.doc_texts).search(" ".join(args.query), args.k):
            print(f"#{h.rank} [{h.doc_id}] {h.title}  ({h.score:.2f})\n    {h.snippet}\n")
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("vilegal.api:app", host=args.host, port=args.port)
    elif args.cmd == "report":
        from vilegal.report import build_report

        print(
            f"report written to {build_report(args.results_dir, args.report_dir, args.kaggle_dir)}"
        )


if __name__ == "__main__":
    main()
