import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIRECTORY = os.path.join(HERE, "result")

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "RAYON_NUM_THREADS")


def child_environment(threads):
    environment = dict(os.environ)
    for variable in THREAD_VARIABLES:
        environment[variable] = str(threads)
    extra_path = os.environ.get("GQP_PROXQP", "")
    if extra_path:
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (extra_path + os.pathsep + existing
                                     if existing else extra_path)
    return environment


def failed_record(set_name, solver_name, problem_path, status, seconds):
    return {"name": os.path.basename(problem_path).split(".")[0],
            "solver": solver_name, "set": set_name,
            "n": None, "m": None, "cones": None,
            "status": status, "seconds": seconds,
            "primal_residual": None, "dual_residual": None,
            "dual_cone_violation": None, "complementarity": None,
            "objective": None, "verified": False,
            "objective_reference": None, "warm_started": False,
            "outer": None, "inner": None}


def write_rows(set_name, solver_name, rows):
    os.makedirs(RESULT_DIRECTORY, exist_ok=True)
    destination = os.path.join(RESULT_DIRECTORY, "%s_%s.json" % (solver_name, set_name))
    with open(destination, "w") as handle:
        json.dump(rows, handle, indent=1)


def sequence_key(benchmark, problem_path):
    getter = getattr(benchmark, "sequence_key", None)
    return getter(problem_path) if getter is not None else None


def run_benchmark(set_name, solver_names, eps_abs, time_limit_seconds, threads,
                  sequential=False):
    benchmark_module = os.path.join(HERE, set_name, set_name + ".py")
    sys.path.insert(0, os.path.join(HERE, "utils"))
    import importlib.util
    specification = importlib.util.spec_from_file_location(set_name + "_set", benchmark_module)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    benchmark = next(getattr(module, name) for name in dir(module)
                     if isinstance(getattr(module, name), type)
                     and getattr(module, name).__module__ == set_name + "_set"
                     and hasattr(getattr(module, name), "list_problems"))

    problem_paths = benchmark.list_problems()
    environment = child_environment(threads)
    rows = {solver_name: [] for solver_name in solver_names}
    runner = os.path.join(HERE, "run_one.py")
    total = len(problem_paths) * len(solver_names)
    done = 0
    started_all = time.perf_counter()

    scratch = os.path.join(RESULT_DIRECTORY, "warm")
    if sequential:
        os.makedirs(scratch, exist_ok=True)
    previous_key = None

    for problem_path in problem_paths:
        key = sequence_key(benchmark, problem_path) if sequential else None
        if sequential and key != previous_key:
            for solver_name in solver_names:
                stale = os.path.join(scratch, "%s_%s.npz" % (solver_name, set_name))
                if os.path.exists(stale):
                    os.remove(stale)
            previous_key = key
        for solver_name in solver_names:
            command = [sys.executable, runner, set_name, problem_path,
                       solver_name, repr(eps_abs)]
            if sequential:
                carried = os.path.join(scratch, "%s_%s.npz" % (solver_name, set_name))
                command += [carried, carried]
            started = time.perf_counter()
            try:
                finished = subprocess.run(command, capture_output=True, text=True,
                                          timeout=time_limit_seconds,
                                          env=environment, cwd=HERE)
                elapsed = time.perf_counter() - started
                lines = (finished.stdout or "").strip().splitlines()
                if finished.returncode != 0 or not lines:
                    record = failed_record(set_name, solver_name, problem_path,
                                           "error", elapsed)
                else:
                    record = json.loads(lines[-1])
            except subprocess.TimeoutExpired:
                record = failed_record(set_name, solver_name, problem_path,
                                       "timeout", float(time_limit_seconds))
            except json.JSONDecodeError:
                record = failed_record(set_name, solver_name, problem_path,
                                       "error", time.perf_counter() - started)
            rows[solver_name].append(record)
            done += 1
            print("[%s] %4d/%d  %-22s %-14s %-10s %s" % (
                set_name, done, total, record["name"], solver_name,
                record["status"], "verified" if record["verified"] else ""),
                flush=True)
        known = next((rows[other][-1] for other in solver_names
                      if rows[other] and rows[other][-1]["n"] is not None), None)
        for solver_name in solver_names:
            latest = rows[solver_name][-1]
            if known is not None and latest["n"] is None:
                latest["n"], latest["m"] = known["n"], known["m"]
                latest["cones"] = known["cones"]
            write_rows(set_name, solver_name, rows[solver_name])

    print("[%s] finished in %.1f s" % (set_name, time.perf_counter() - started_all))
    for solver_name in solver_names:
        verified = sum(1 for row in rows[solver_name] if row["verified"])
        print("   %-14s %3d verified of %d" % (solver_name, verified, len(problem_paths)))
