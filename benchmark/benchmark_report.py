import concurrent.futures
import importlib.util
import inspect
import json
import os
import queue
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "utils"))

import run_store
from bench import (RESULT_DIRECTORY, child_environment, failed_record,
                   sequence_key, write_rows)
from aggregate import sgm_seconds, solved_count

SOLVERS = ["proxgqp_interior", "proxgqp_semismooth", "clarabel", "piqp", "proxqp"]
EPS_ABS = 1e-9
TIME_LIMIT_SECONDS = 20
OUT_OF_THE_BOX = os.environ.get("GQP_OUT_OF_THE_BOX", "") == "1"
WORKERS = max(1, (os.cpu_count() or 8) - 4)

SETS = ["random_qp", "random_socp", "random_psd", "random_exp", "random_pow",
        "random_mixed", "maros", "netlib", "cblib",
        "skfolio_real_qp", "skfolio_real_curved",
        "skfolio_random_qp", "skfolio_random_curved",
        "seq_rebalance", "seq_rebalance_curved"]
SEQUENTIAL_SETS = {"seq_rebalance", "seq_rebalance_curved"}


def load_benchmark(set_name):
    path = os.path.join(HERE, set_name, set_name + ".py")
    specification = importlib.util.spec_from_file_location(set_name + "_set", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for _name, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ == set_name + "_set" and hasattr(candidate, "list_problems"):
            return candidate
    raise ImportError("no benchmark class in " + path)


class CorePool:

    def __init__(self, count):
        self.available = queue.Queue()
        for index in range(count):
            self.available.put(index)

    def acquire(self):
        return self.available.get()

    def release(self, core):
        self.available.put(core)


def run_one(set_name, problem_path, solver_name, cores, environment,
            carried=None):
    core = cores.acquire()
    command = ["taskset", "-c", str(core), sys.executable,
               os.path.join(HERE, "run_one.py"), set_name, problem_path,
               solver_name, repr(EPS_ABS)]
    if carried is not None:
        command += [carried, carried]
    started = time.perf_counter()
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=TIME_LIMIT_SECONDS,
                                  env=environment, cwd=HERE)
        elapsed = time.perf_counter() - started
        lines = (finished.stdout or "").strip().splitlines()
        if finished.returncode != 0 or not lines:
            return failed_record(set_name, solver_name, problem_path, "error", elapsed)
        return json.loads(lines[-1])
    except subprocess.TimeoutExpired:
        return failed_record(set_name, solver_name, problem_path, "timeout",
                             float(TIME_LIMIT_SECONDS))
    except json.JSONDecodeError:
        return failed_record(set_name, solver_name, problem_path, "error",
                             time.perf_counter() - started)
    finally:
        cores.release(core)


def run_parallel_set(set_name, benchmark, cores, environment):
    problem_paths = benchmark.list_problems()
    rows = {solver_name: [] for solver_name in SOLVERS}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {}
        for problem_path in problem_paths:
            for solver_name in SOLVERS:
                future = pool.submit(run_one, set_name, problem_path,
                                     solver_name, cores, environment)
                pending[future] = solver_name
        for future in concurrent.futures.as_completed(pending):
            rows[pending[future]].append(future.result())
    return rows


def run_sequential_set(set_name, benchmark, cores, environment):
    problem_paths = benchmark.list_problems()
    chains = {}
    for problem_path in problem_paths:
        chains.setdefault(sequence_key(benchmark, problem_path), []).append(problem_path)

    scratch = os.path.join(RESULT_DIRECTORY, "warm")
    os.makedirs(scratch, exist_ok=True)
    rows = {solver_name: [] for solver_name in SOLVERS}

    def run_chain(key, paths, solver_name):
        carried = os.path.join(scratch, "%s_%s_%s.npz" % (solver_name, set_name, key))
        if os.path.exists(carried):
            os.remove(carried)
        produced = []
        for problem_path in paths:
            produced.append(run_one(set_name, problem_path, solver_name, cores,
                                    environment, carried))
        return solver_name, produced

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(run_chain, key, paths, solver_name)
                   for key, paths in chains.items()
                   for solver_name in SOLVERS]
        for future in concurrent.futures.as_completed(pending):
            solver_name, produced = future.result()
            rows[solver_name].extend(produced)
    return rows


def expressible(entries):
    return [row for row in entries if row.get("status") != "unsupported"]


def report(all_rows):
    print("")
    print("=" * 78)
    for set_name in SETS:
        if set_name not in all_rows:
            continue
        rows = all_rows[set_name]
        print("")
        print("  %s" % set_name)
        print("  %-22s %10s %14s %10s" % ("solver", "verified",
                                            "sgm seconds", "skipped"))
        for solver_name in SOLVERS:
            raw = rows.get(solver_name, [])
            entries = expressible(raw)
            if not entries:
                print("  %-22s %10s %14s %10d"
                      % (solver_name, "-", "-", len(raw)))
                continue
            verified, attempted = solved_count(entries)
            print("  %-22s %5d/%-4d %14.4f %10d"
                  % (solver_name, verified, attempted,
                     sgm_seconds(entries, TIME_LIMIT_SECONDS),
                     len(raw) - len(entries)))

    print("")
    print("=" * 78)
    print("  overall, every set pooled")
    print("  %-22s %10s %14s %10s" % ("solver", "verified", "sgm seconds",
                                       "skipped"))
    for solver_name in SOLVERS:
        raw = [row for rows in all_rows.values() for row in rows.get(solver_name, [])]
        pooled = expressible(raw)
        if not pooled:
            continue
        verified, attempted = solved_count(pooled)
        print("  %-22s %5d/%-4d %14.4f %10d"
              % (solver_name, verified, attempted,
                 sgm_seconds(pooled, TIME_LIMIT_SECONDS),
                 len(raw) - len(pooled)))


def main():
    environment = child_environment(1, OUT_OF_THE_BOX)
    cores = CorePool(WORKERS)
    all_rows = {}
    started_all = time.perf_counter()
    run_directory = run_store.create(
        SETS, SOLVERS,
        {"eps_abs": EPS_ABS, "time_limit_seconds": TIME_LIMIT_SECONDS,
         "threads": 1, "parallel_workers": WORKERS,
         "out_of_the_box": OUT_OF_THE_BOX})
    print("  run %s" % os.path.basename(run_directory), flush=True)
    print("  %d workers, %.0fs cap%s, solvers: %s"
          % (WORKERS, TIME_LIMIT_SECONDS,
             ", out of the box" if OUT_OF_THE_BOX else "",
             ", ".join(SOLVERS)), flush=True)

    for set_name in SETS:
        try:
            benchmark = load_benchmark(set_name)
        except Exception as error:
            print("  %-16s SKIPPED %s" % (set_name, str(error)[:60]), flush=True)
            continue
        started = time.perf_counter()
        runner = run_sequential_set if set_name in SEQUENTIAL_SETS else run_parallel_set
        rows = runner(set_name, benchmark, cores, environment)
        all_rows[set_name] = rows
        for solver_name, entries in rows.items():
            write_rows(run_directory, set_name, solver_name, entries)
        counts = " ".join("%s %d/%d" % ((name.replace("proxgqp_", "gqp_"),)
                                        + solved_count(entries))
                          for name, entries in rows.items() if entries)
        print("  %-16s %7.1fs   %s" % (set_name, time.perf_counter() - started,
                                       counts), flush=True)

    run_store.finish(run_directory, started_all,
                     {"sets_completed": sorted(all_rows)})
    report(all_rows)
    print("")
    print("  run %s" % os.path.basename(run_directory))
    print("  total %.1f minutes" % ((time.perf_counter() - started_all) / 60.0))


if __name__ == "__main__":
    main()
