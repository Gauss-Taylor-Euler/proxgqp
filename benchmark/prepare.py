import argparse
import gzip
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "utils"))

MANIFEST_PATH = os.path.join(HERE, "manifest.json")

GENERATED_SETS = ("random_qp", "random_socp", "random_psd", "random_exp",
                  "random_pow", "random_mixed", "skfolio_real",
                  "skfolio_random", "seq_rebalance")

COLLECTED_SETS = {
    "maros": {"suffix": ".mat",
              "source": "https://github.com/qpsolvers/qpbenchmark",
              "licence": "freely redistributable, Maros and Meszaros 1999"},
    "netlib": {"suffix": ".mps",
               "source": "https://netlib.org/lp/data/",
               "licence": "public domain"},
    "cblib": {"suffix": ".cbf.gz",
              "source": "https://cblib.zib.de/download/all/",
              "licence": "see cblib.zib.de terms, redistribution not assumed"},
}

MARKET_DIRECTORY = os.path.join(HERE, "market_data")


def digest_of(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_benchmark(set_name):
    path = os.path.join(HERE, set_name, set_name + ".py")
    specification = importlib.util.spec_from_file_location(set_name + "_set", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for _name, candidate in inspect.getmembers(module, inspect.isclass):
        if candidate.__module__ == set_name + "_set" and hasattr(candidate, "list_problems"):
            return candidate
    raise ImportError("no benchmark class in " + path)


def problem_digest(problem):
    import numpy
    import scipy.sparse
    hasher = hashlib.sha256()
    objective = scipy.sparse.csc_matrix(problem["P"])
    constraint = scipy.sparse.csc_matrix(problem["E"])
    for array in (objective.indptr, objective.indices, objective.data,
                  numpy.asarray(problem["q"], float),
                  constraint.indptr, constraint.indices, constraint.data,
                  numpy.asarray(problem["b"], float)):
        hasher.update(numpy.ascontiguousarray(array).tobytes())
    hasher.update(repr(problem["cones"]).encode())
    return hasher.hexdigest()


def generate(sample):
    print("  regenerating the seeded sets and checking they are reproducible")
    failures = 0
    for set_name in GENERATED_SETS:
        try:
            benchmark = load_benchmark(set_name)
        except Exception as error:
            print("    %-16s SKIPPED  %s" % (set_name, str(error)[:60]))
            failures += 1
            continue
        identifiers = benchmark.list_problems()
        chosen = identifiers[:sample] if sample else identifiers
        mismatched = 0
        for identifier in chosen:
            first = problem_digest(benchmark.load(identifier))
            second = problem_digest(benchmark.load(identifier))
            if first != second:
                mismatched += 1
                print("    %-16s NOT REPRODUCIBLE  %s" % (set_name, identifier))
        state = "ok" if mismatched == 0 else "%d mismatched" % mismatched
        failures += mismatched
        print("    %-16s %4d problems, %d checked, %s"
              % (set_name, len(identifiers), len(chosen), state))
    return failures


def collected_files(set_name):
    directory = os.path.join(HERE, set_name, "data")
    if not os.path.isdir(directory):
        return []
    suffix = COLLECTED_SETS[set_name]["suffix"]
    return sorted(name for name in os.listdir(directory) if name.endswith(suffix))


def write_manifest():
    manifest = {}
    for set_name, description in COLLECTED_SETS.items():
        directory = os.path.join(HERE, set_name, "data")
        entries = {}
        for name in collected_files(set_name):
            path = os.path.join(directory, name)
            entries[name] = {"sha256": digest_of(path),
                             "bytes": os.path.getsize(path)}
        manifest[set_name] = {"source": description["source"],
                              "licence": description["licence"],
                              "files": entries}
        print("    %-8s %4d files" % (set_name, len(entries)))
    market = {}
    if os.path.isdir(MARKET_DIRECTORY):
        for name in sorted(os.listdir(MARKET_DIRECTORY)):
            path = os.path.join(MARKET_DIRECTORY, name)
            if os.path.isfile(path):
                market[name] = {"sha256": digest_of(path),
                                "bytes": os.path.getsize(path)}
    manifest["market_data"] = {"source": "committed with the repository",
                               "licence": "derived from skfolio, BSD-3",
                               "files": market}
    print("    %-8s %4d files" % ("market", len(market)))
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    print("  manifest written to %s" % os.path.relpath(MANIFEST_PATH, HERE))
    return 0


def read_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print("  no manifest at %s; run 'prepare.py checksum' from a tree that "
              "already holds the data" % MANIFEST_PATH)
        return None
    with open(MANIFEST_PATH) as handle:
        return json.load(handle)


def verify():
    manifest = read_manifest()
    if manifest is None:
        return 1
    failures = 0
    for set_name in sorted(manifest):
        expected = manifest[set_name]["files"]
        directory = (MARKET_DIRECTORY if set_name == "market_data"
                     else os.path.join(HERE, set_name, "data"))
        missing, corrupt = [], []
        for name, record in expected.items():
            path = os.path.join(directory, name)
            if not os.path.exists(path):
                missing.append(name)
            elif digest_of(path) != record["sha256"]:
                corrupt.append(name)
        failures += len(missing) + len(corrupt)
        state = "ok" if not missing and not corrupt else \
            "%d missing, %d corrupt" % (len(missing), len(corrupt))
        print("    %-12s %4d expected, %s" % (set_name, len(expected), state))
        for name in (missing + corrupt)[:5]:
            print("        %s" % name)
    return failures


def fetch(set_names):
    manifest = read_manifest()
    if manifest is None:
        return 1
    failures = 0
    for set_name in set_names:
        description = COLLECTED_SETS[set_name]
        directory = os.path.join(HERE, set_name, "data")
        os.makedirs(directory, exist_ok=True)
        expected = manifest.get(set_name, {}).get("files", {})
        if not expected:
            print("    %-8s no files listed in the manifest" % set_name)
            failures += 1
            continue
        wanted = [name for name in expected
                  if not os.path.exists(os.path.join(directory, name))
                  or digest_of(os.path.join(directory, name)) != expected[name]["sha256"]]
        if not wanted:
            print("    %-8s %4d files already present and verified"
                  % (set_name, len(expected)))
            continue
        print("    %-8s %d of %d files to fetch from %s"
              % (set_name, len(wanted), len(expected), description["source"]))
        print("        this collection has no per-file download endpoint wired "
              "in; obtain the archive from the source above, unpack it into %s, "
              "then run 'prepare.py verify'"
              % os.path.relpath(directory, HERE))
        print("        licence: %s" % description["licence"])
        failures += len(wanted)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="prepare the benchmark data")
    parser.add_argument("action", nargs="?", default="all",
                        choices=("all", "generate", "fetch", "verify", "checksum"))
    parser.add_argument("--sample", type=int, default=4,
                        help="problems per generated set to check, 0 for all")
    parser.add_argument("--sets", nargs="*", default=sorted(COLLECTED_SETS))
    arguments = parser.parse_args(argv)

    failures = 0
    if arguments.action in ("all", "generate"):
        failures += generate(arguments.sample)
    if arguments.action == "checksum":
        print("  hashing the collected data")
        failures += write_manifest()
    if arguments.action in ("all", "fetch"):
        print("  collected sets")
        failures += fetch(arguments.sets)
    if arguments.action in ("all", "verify"):
        print("  verifying against the manifest")
        failures += verify()

    print("")
    if failures:
        print("  %d problem(s); the benchmark is not ready to run" % failures)
    else:
        print("  everything present and verified")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
