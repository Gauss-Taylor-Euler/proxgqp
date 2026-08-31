from . import _core

def _methods():
    table = {"semismooth": _core.Method.semismooth,
             "interior": _core.Method.interior,
             "interior_exp": _core.Method.interior_exp,
             "barrier": _core.Method.interior,
             "projection": _core.Method.semismooth}
    return table


METHOD = _methods()
LINE_SEARCH = {"exact": _core.LineSearch.exact, "armijo": _core.LineSearch.armijo,
               "decrease": _core.LineSearch.decrease}
PENALTY = {"bcl": _core.Penalty.bcl, "gbcl": _core.Penalty.gbcl,
           "gbcl_exp": _core.Penalty.gbcl_exp}
ROAD = {"auto": _core.Road.automatic,
        "three_by_three": _core.Road.three_by_three, "3x3": _core.Road.three_by_three,
        "schur": _core.Road.schur}
BACKEND = {"eigen_sparse": _core.Backend.eigen_sparse,
           "eigen_dense": _core.Backend.eigen_dense,
           "qdldl": _core.Backend.qdldl, "cholmod": _core.Backend.cholmod,
           "lapack_dense": _core.Backend.lapack_dense,
           "auto": _core.Backend.automatic}

NAMED = {"method": METHOD, "line_search": LINE_SEARCH, "penalty": PENALTY,
         "road": ROAD, "backend": BACKEND}

PRESETS = {
    "default": {},
    "interior": {"method": "interior"},
    "semismooth": {"method": "semismooth"},
    "accurate": {"eps_abs": 1e-12, "eps_rel": 1e-12},
    "loose": {"eps_abs": 1e-6, "eps_rel": 1e-6},
    "no_equilibration": {"equilibrate": False},
    "no_mehrotra": {"interior.mehrotra": False},
    "schur": {"road": "schur"},
    "verbose": {"verbose": True},
    "armijo": {"line_search": "armijo"},
    "decrease": {"line_search": "decrease"},
    "bcl": {"penalty": "bcl"},
}


def _targets(settings):
    found = [("", settings), ("bcl", settings.tuning.bcl),
             ("gbcl", settings.tuning.gbcl),
             ("interior", settings.tuning.interior),
             ("interior_exp", settings.tuning.interior_exp),
             ("gbcl_exp", settings.tuning.gbcl_exp),
             ("semismooth", settings.tuning.semismooth)]
    return tuple(found)


def knob_names():
    out = []
    for prefix, target in _targets(None) if False else ():
        pass
    settings = _core.Settings()
    for prefix, target in _targets(settings):
        for name in dir(target):
            if name.startswith("_") or name == "tuning":
                continue
            out.append("%s.%s" % (prefix, name) if prefix else name)
    return sorted(set(out))


def resolve(key, value):
    table = NAMED.get(key.rsplit(".", 1)[-1])
    if table is None or not isinstance(value, str):
        return value
    try:
        return table[value.lower()]
    except KeyError:
        raise ValueError("%s has no value %r; use one of %s"
                         % (key, value, sorted(table))) from None


def apply(settings, key, value):
    """Set one knob.  A dotted key names a group outright, as in
    interior.tau; a bare key searches the groups, the method in force first."""
    resolved = resolve(key, value)
    if "." in key:
        head, tail = key.split(".", 1)
        for prefix, target in _targets(settings):
            if prefix == head and hasattr(target, tail):
                setattr(target, tail, resolved)
                return
        raise ValueError("no knob %r" % key)
    for _prefix, target in _targets(settings):
        if hasattr(target, key):
            setattr(target, key, resolved)
            return
    raise ValueError("no knob %r; use one of %s" % (key, knob_names()))


def make_settings(preset=None, **knobs):
    settings = _core.Settings()
    merged = {}
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError("no preset %r; use one of %s" % (preset, sorted(PRESETS)))
        merged.update(PRESETS[preset])
    merged.update(knobs)
    for key, value in merged.items():
        apply(settings, key, value)
    return settings


def describe(settings):
    out = {}
    for prefix, target in _targets(settings):
        for name in dir(target):
            if name.startswith("_") or name == "tuning":
                continue
            value = getattr(target, name)
            out["%s.%s" % (prefix, name) if prefix else name] = value
    return out
