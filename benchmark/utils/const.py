import os

UNBOUNDED_ITERATIONS = 10_000_000
UNBOUNDED_SECONDS = 1e12
INFINITY = 1e30
SMALLEST_RELATIVE_TOLERANCE = 1e-9
REGULARIZATION_BELOW_TOLERANCE = 1e-2

OUT_OF_THE_BOX = os.environ.get("GQP_OUT_OF_THE_BOX", "") == "1"


def imposed(**settings):
    """The accuracy and iteration settings the benchmark imposes on a solver.

    Empty when the run is out of the box, so a solver is left on its own
    defaults.  Thread counts and output flags are not routed through this: a
    solver that picks its own width is not being measured on the same machine
    as the others, and a solver that prints breaks the runner's protocol.
    """
    return {} if OUT_OF_THE_BOX else settings
