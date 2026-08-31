import importlib.util
import json
import os
import sys
import time

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "utils"))
sys.path.insert(0, os.path.join(HERE, "solvers"))

from verify import verify
import proxgqp_interior as adapter

_cache = {}


def load_set(set_name, limit=None):
    key = (set_name, limit)
    if key in _cache:
        return _cache[key]
    specification = importlib.util.spec_from_file_location(
        set_name + "_set", os.path.join(HERE, set_name, set_name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    cls = next(getattr(module, n) for n in dir(module)
               if isinstance(getattr(module, n), type)
               and getattr(module, n).__module__ == set_name + "_set"
               and hasattr(getattr(module, n), "list_problems"))
    names = cls.list_problems()
    if limit:
        names = names[:limit]
    problems = []
    for name in names:
        try:
            problems.append((name, cls.load(name)))
        except Exception:
            pass
    _cache[key] = problems
    return problems


def score(set_name, eps_abs=1e-9, smoothing=0, limit=None, time_limit=20.0,
          **overrides):
    problems = load_set(set_name, limit)
    settings = adapter.make_settings(eps_abs, smoothing, **overrides)
    good = 0
    total = 0.0
    failures = []
    for name, problem in problems:
        started = time.perf_counter()
        try:
            results = adapter.run(problem, settings)
            payload = {"status": results.status,
                       "x": numpy.asarray(results.x),
                       "z": numpy.asarray(results.z)}
        except Exception:
            failures.append((name, "exception"))
            total += time_limit
            continue
        elapsed = time.perf_counter() - started
        total += min(elapsed, time_limit)
        try:
            report = verify(problem, payload, eps_abs)
        except Exception:
            failures.append((name, "verify"))
            continue
        if report.get("verified"):
            good += 1
        else:
            failures.append((name, "%.1e" % max(
                report["primal_residual"], report["dual_residual"],
                report["dual_cone_violation"], report["complementarity"])))
    return {"set": set_name, "n": len(problems), "verified": good,
            "seconds": total, "failures": failures}


def show(label, result):
    print("  %-30s %3d/%-3d verified  %7.2fs" % (
        label, result["verified"], result["n"], result["seconds"]))
    return result


def iterations(set_name, eps_abs=1e-9, smoothing=0, limit=None, **overrides):
    problems = load_set(set_name, limit)
    settings = adapter.make_settings(eps_abs, smoothing, **overrides)
    good = 0
    total_inner = 0
    total_time = 0.0
    for name, problem in problems:
        started = time.perf_counter()
        try:
            r = adapter.run(problem, settings)
        except Exception:
            total_inner += 1000
            continue
        total_time += time.perf_counter() - started
        total_inner += r.inner_iterations
        payload = {"status": r.status,
                   "x": numpy.asarray(r.x), "z": numpy.asarray(r.z)}
        try:
            if verify(problem, payload, eps_abs).get("verified"):
                good += 1
        except Exception:
            pass
    return good, total_inner, total_time, len(problems)
