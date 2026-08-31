import os

import numpy
import scipy.io
import scipy.sparse


class Maros:

    name = "maros"
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    equality_tolerance = 1e-12
    infinity = 1e19

    @classmethod
    def list_problems(cls):
        return sorted(
            os.path.join(cls.directory, filename)
            for filename in os.listdir(cls.directory)
            if filename.endswith(".mat")
        )

    @classmethod
    def load(cls, path):
        contents = scipy.io.loadmat(path)
        objective_hessian = scipy.sparse.csc_matrix(contents["P"])
        objective_gradient = numpy.asarray(contents["q"]).ravel().astype(float)
        constraint_matrix = scipy.sparse.csc_matrix(contents["A"])
        lower_bound = numpy.asarray(contents["l"]).ravel().astype(float)
        upper_bound = numpy.asarray(contents["u"]).ravel().astype(float)

        is_equality = numpy.abs(upper_bound - lower_bound) <= cls.equality_tolerance
        is_inequality = ~is_equality

        blocks = []
        right_hand_sides = []
        cones = []

        if is_equality.any():
            blocks.append(constraint_matrix[is_equality])
            right_hand_sides.append(upper_bound[is_equality])
            cones.append(("Zero", int(is_equality.sum()), {}))

        inequality_rows = []
        inequality_values = []

        bounded_above = upper_bound[is_inequality] < cls.infinity
        if bounded_above.any():
            inequality_rows.append(constraint_matrix[is_inequality][bounded_above])
            inequality_values.append(upper_bound[is_inequality][bounded_above])

        bounded_below = lower_bound[is_inequality] > -cls.infinity
        if bounded_below.any():
            inequality_rows.append(-constraint_matrix[is_inequality][bounded_below])
            inequality_values.append(-lower_bound[is_inequality][bounded_below])

        if inequality_rows:
            stacked = scipy.sparse.vstack(inequality_rows).tocsc()
            blocks.append(stacked)
            right_hand_sides.append(numpy.concatenate(inequality_values))
            cones.append(("Nonneg", stacked.shape[0], {}))

        if not blocks:
            raise ValueError("no constraints")

        return {
            "P": objective_hessian,
            "q": objective_gradient,
            "E": scipy.sparse.vstack(blocks).tocsc(),
            "f": numpy.concatenate(right_hand_sides),
            "cones": cones,
        }
