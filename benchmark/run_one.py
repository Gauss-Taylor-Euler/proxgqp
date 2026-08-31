import importlib.util
import inspect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "utils"))


def load_class(module_path, module_name, required_attribute):
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for _name, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ == module_name and hasattr(candidate, required_attribute):
            return candidate
    raise ImportError("no class with %r in %s" % (required_attribute, module_path))


def emit(record):
    print(json.dumps(record))


def main(argv):
    set_name, problem_path, solver_name, tolerance = argv[:4]
    warm_start_file = argv[4] if len(argv) > 4 else ""
    solution_file = argv[5] if len(argv) > 5 else ""
    eps_abs = float(tolerance)

    benchmark = load_class(os.path.join(HERE, set_name, set_name + ".py"),
                           set_name + "_set", "list_problems")
    solver = load_class(os.path.join(HERE, "solvers", solver_name + ".py"),
                        solver_name + "_solver", "solve")

    from verify import verify

    record = {"name": os.path.basename(problem_path).split(".")[0],
              "solver": solver_name, "set": set_name,
              "n": None, "m": None, "cones": None,
              "status": "error", "seconds": None,
              "primal_residual": None, "dual_residual": None,
              "dual_cone_violation": None, "complementarity": None,
              "objective": None, "verified": False,
              "objective_reference": None, "warm_started": False,
              "outer": None, "inner": None}

    problem = benchmark.load(problem_path)
    record["n"] = int(problem["P"].shape[0])
    record["m"] = int(problem["E"].shape[0])
    record["cones"] = sorted({kind for kind, _size, _params in problem["cones"]})

    accepts_quadratic = getattr(solver, "quadratic", True)
    if (not set(record["cones"]).issubset(set(solver.cones))
            or (problem["P"].nnz > 0 and not accepts_quadratic)):
        record["status"] = "unsupported"
        emit(record)
        return 0

    import numpy

    warm_start = None
    if warm_start_file and os.path.exists(warm_start_file):
        stored = numpy.load(warm_start_file)
        if (stored["x"].size == record["n"] and stored["z"].size == record["m"]):
            warm_start = {"x": stored["x"], "z": stored["z"]}
    record["warm_started"] = warm_start is not None

    started = time.perf_counter()
    result = solver.solve(problem, eps_abs, warm_start)
    record["seconds"] = time.perf_counter() - started
    if solution_file and result.get("x") is not None and result.get("z") is not None:
        numpy.savez(solution_file, x=numpy.asarray(result["x"], float),
                    z=numpy.asarray(result["z"], float))
    record["status"] = result["status"]
    record["outer"] = result.get("outer")
    record["inner"] = result.get("inner")
    record.update(verify(problem, result, eps_abs))
    record["objective_reference"] = problem.get("objective_reference")
    emit(record)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
