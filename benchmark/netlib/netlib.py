import os

import numpy
import scipy.sparse


class Netlib:

    name = "netlib"
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    equality_tolerance = 1e-12
    infinity = 1e19

    @classmethod
    def list_problems(cls):
        return sorted(
            os.path.join(cls.directory, filename)
            for filename in os.listdir(cls.directory)
            if filename.endswith(".mps")
        )

    @classmethod
    def load(cls, path):
        import highspy

        reader = highspy.Highs()
        reader.setOptionValue("output_flag", False)
        reader.readModel(path)
        model = reader.getLp()

        variable_count = model.num_col_
        row_count = model.num_row_
        if variable_count == 0:
            raise ValueError("empty model")

        constraint_matrix = scipy.sparse.csc_matrix(
            (numpy.array(model.a_matrix_.value_),
             numpy.array(model.a_matrix_.index_),
             numpy.array(model.a_matrix_.start_)),
            shape=(row_count, variable_count))

        objective_gradient = numpy.array(model.col_cost_, float)
        row_lower = numpy.array(model.row_lower_, float)
        row_upper = numpy.array(model.row_upper_, float)
        variable_lower = numpy.array(model.col_lower_, float)
        variable_upper = numpy.array(model.col_upper_, float)

        is_equality = numpy.abs(row_upper - row_lower) <= cls.equality_tolerance
        is_inequality = ~is_equality

        blocks = []
        right_hand_sides = []
        cones = []

        if is_equality.any():
            blocks.append(constraint_matrix[is_equality])
            right_hand_sides.append(row_upper[is_equality])
            cones.append(("Zero", int(is_equality.sum()), {}))

        inequality_rows = []
        inequality_values = []
        identity = scipy.sparse.eye(variable_count, format="csc")

        row_bounded_above = row_upper[is_inequality] < cls.infinity
        if row_bounded_above.any():
            inequality_rows.append(constraint_matrix[is_inequality][row_bounded_above])
            inequality_values.append(row_upper[is_inequality][row_bounded_above])

        row_bounded_below = row_lower[is_inequality] > -cls.infinity
        if row_bounded_below.any():
            inequality_rows.append(-constraint_matrix[is_inequality][row_bounded_below])
            inequality_values.append(-row_lower[is_inequality][row_bounded_below])

        variable_bounded_above = variable_upper < cls.infinity
        if variable_bounded_above.any():
            inequality_rows.append(identity[variable_bounded_above])
            inequality_values.append(variable_upper[variable_bounded_above])

        variable_bounded_below = variable_lower > -cls.infinity
        if variable_bounded_below.any():
            inequality_rows.append(-identity[variable_bounded_below])
            inequality_values.append(-variable_lower[variable_bounded_below])

        if inequality_rows:
            stacked = scipy.sparse.vstack(inequality_rows).tocsc()
            blocks.append(stacked)
            right_hand_sides.append(numpy.concatenate(inequality_values))
            cones.append(("Nonneg", stacked.shape[0], {}))

        if not blocks:
            raise ValueError("no constraints")

        return {
            "P": scipy.sparse.csc_matrix((variable_count, variable_count)),
            "q": objective_gradient,
            "E": scipy.sparse.vstack(blocks).tocsc(),
            "f": numpy.concatenate(right_hand_sides),
            "cones": cones,
        }
