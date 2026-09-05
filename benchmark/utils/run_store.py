"""Storage layout for benchmark runs.

Every run owns a directory under result/runs, named by a sortable timestamp
and a short unique suffix, holding one file per solver and set plus a
meta.json recording when the run started, which sets and solvers it covered,
the settings it used and the solver versions.  Reading a number back later is
otherwise guesswork: a bare solver_set.json says nothing about when it was
produced, at what time limit, or against which version of a rival solver.
"""
import json
import os
import platform
import sys
import time
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIRECTORY = os.path.join(HERE, "result")
RUNS_DIRECTORY = os.path.join(RESULT_DIRECTORY, "runs")
LATEST_LINK = os.path.join(RUNS_DIRECTORY, "latest")
META_NAME = "meta.json"

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "RAYON_NUM_THREADS")


def new_identifier():
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return "%s-%s" % (stamp, uuid.uuid4().hex[:8])


DISTRIBUTIONS = {"proxqp": "proxsuite"}


def solver_versions(names):
    import importlib.metadata as metadata
    out = {}
    for name in names:
        package = name.split("_")[0]
        try:
            out[name] = metadata.version(DISTRIBUTIONS.get(package, package))
        except Exception:
            out[name] = None
    return out


def machine():
    return {"processor": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES}}


def create(sets, solvers, settings, identifier=None):
    identifier = identifier or new_identifier()
    directory = os.path.join(RUNS_DIRECTORY, identifier)
    os.makedirs(directory, exist_ok=True)
    meta = {"run_id": identifier,
            "uid": identifier.split("-")[-1],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "finished_utc": None,
            "duration_seconds": None,
            "sets": list(sets),
            "solvers": list(solvers),
            "settings": dict(settings),
            "machine": machine(),
            "versions": solver_versions(solvers),
            "complete": False}
    write_meta(directory, meta)
    _point_latest_at(directory)
    return directory


def write_meta(directory, meta):
    with open(os.path.join(directory, META_NAME), "w") as handle:
        json.dump(meta, handle, indent=1, sort_keys=True)


def read_meta(directory):
    try:
        with open(os.path.join(directory, META_NAME)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def finish(directory, started_at, extra=None):
    meta = read_meta(directory)
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["duration_seconds"] = round(time.perf_counter() - started_at, 3)
    meta["complete"] = True
    if extra:
        meta.update(extra)
    write_meta(directory, meta)
    return meta


def _point_latest_at(directory):
    try:
        if os.path.islink(LATEST_LINK) or os.path.exists(LATEST_LINK):
            os.remove(LATEST_LINK)
        os.symlink(os.path.basename(directory), LATEST_LINK)
    except OSError:
        pass


def file_name(set_name, solver_name):
    return "%s_%s.json" % (solver_name, set_name)


def write_rows(directory, set_name, solver_name, rows):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, file_name(set_name, solver_name)), "w") as handle:
        json.dump(rows, handle, indent=1)


def runs():
    if not os.path.isdir(RUNS_DIRECTORY):
        return []
    found = [name for name in os.listdir(RUNS_DIRECTORY)
             if name != "latest"
             and os.path.isdir(os.path.join(RUNS_DIRECTORY, name))]
    return sorted(found)


def resolve(specification):
    if specification is None:
        return None
    if os.path.isdir(specification):
        return specification
    candidate = os.path.join(RUNS_DIRECTORY, specification)
    if os.path.isdir(candidate):
        return candidate
    if specification == "latest":
        available = runs()
        if available:
            return os.path.join(RUNS_DIRECTORY, available[-1])
    raise ValueError("no such run: %s" % specification)


def load_rows(set_name, solver_name, run=None):
    """Rows for one solver and set, from a run directory or the flat layout."""
    places = []
    if run is not None:
        places.append(os.path.join(resolve(run), file_name(set_name, solver_name)))
    else:
        for name in reversed(runs()):
            places.append(os.path.join(RUNS_DIRECTORY, name,
                                       file_name(set_name, solver_name)))
    places.append(os.path.join(RESULT_DIRECTORY, file_name(set_name, solver_name)))
    for place in places:
        if os.path.exists(place):
            with open(place) as handle:
                return json.load(handle)
    return None
