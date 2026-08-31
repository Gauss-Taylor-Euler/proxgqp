import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "utils"))

from bench import run_benchmark

SET_NAME = "skfolio_random_qp"
SOLVERS = ["proxgqp_interior", "proxgqp_semismooth", "clarabel", "osqp", "piqp", "proxqp", "scs", "ecos", "highs_simplex", "highs_ipm"]
EPS_ABS = 1e-9
TIME_LIMIT_SECONDS = 20
THREADS = 1

if __name__ == "__main__":
    run_benchmark(SET_NAME, SOLVERS, EPS_ABS, TIME_LIMIT_SECONDS, THREADS)
