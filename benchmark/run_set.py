import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from bench import run_benchmark

import benchmark_report

parser = argparse.ArgumentParser(description="run one benchmark set")
parser.add_argument("set_name", choices=benchmark_report.SETS)
parser.add_argument("--solvers", nargs="+", default=benchmark_report.SOLVERS)
parser.add_argument("--eps", type=float, default=benchmark_report.EPS_ABS,
                    help="tolerance asked of each solver, and the tolerance "
                         "the result is verified at; with --out-of-the-box "
                         "no solver is told it, so it is the verification "
                         "tolerance alone")
parser.add_argument("--limit", type=float,
                    default=benchmark_report.TIME_LIMIT_SECONDS)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--out-of-the-box", action="store_true",
                    help="leave every solver on its own defaults; only the "
                         "thread count is pinned")
parser.add_argument("--run", default=None,
                    help="append to an existing run id, or 'latest'; "
                         "omit to start a new one")

if __name__ == "__main__":
    given = parser.parse_args()
    run_benchmark(given.set_name, given.solvers, given.eps, given.limit,
                  given.threads,
                  sequential=given.set_name in benchmark_report.SEQUENTIAL_SETS,
                  run=given.run,
                  out_of_the_box=given.out_of_the_box
                  or benchmark_report.OUT_OF_THE_BOX)
