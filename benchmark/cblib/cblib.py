import gzip
import math
import os

import numpy
import scipy.sparse


class Unsupported(Exception):
    pass


class Cblib:

    name = "cblib"
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    known_cones = ("F", "L=", "L+", "L-", "Q", "QR", "EXP", "POW")

    @classmethod
    def list_problems(cls):
        return sorted(
            os.path.join(cls.directory, filename)
            for filename in os.listdir(cls.directory)
            if filename.endswith(".cbf.gz")
        )

    @staticmethod
    def read_tokens(path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    yield line

    @staticmethod
    def svec_length(side):
        return side * (side + 1) // 2

    @staticmethod
    def svec_index(side, matrix_row, matrix_column):
        if matrix_row < matrix_column:
            matrix_row, matrix_column = matrix_column, matrix_row
        return (matrix_column * side
                - matrix_column * (matrix_column - 1) // 2
                + (matrix_row - matrix_column))

    @staticmethod
    def split_cone_name(name):
        if name.startswith("@"):
            table_index, kind = name[1:].split(":")
            return kind, int(table_index)
        return name, None

    @staticmethod
    def rotation_for_rotated_cone(dimension):
        rotation = scipy.sparse.lil_matrix((dimension, dimension))
        inverse_root_two = 1.0 / math.sqrt(2.0)
        rotation[0, 0] = inverse_root_two
        rotation[0, 1] = inverse_root_two
        rotation[1, 0] = inverse_root_two
        rotation[1, 1] = -inverse_root_two
        for position in range(2, dimension):
            rotation[position, position] = 1.0
        return rotation.tocsc()

    @staticmethod
    def power_exponent(table_index, parsed):
        table = parsed["power_cones"]
        if not table:
            raise Unsupported("POW without a cone table")
        exponents = table[table_index]
        if len(exponents) != 2:
            raise Unsupported("POW of order %d" % len(exponents))
        exponent_sum = exponents[0] + exponents[1]
        if exponent_sum <= 0:
            raise Unsupported("degenerate POW exponents")
        return exponents[0] / exponent_sum

    @classmethod
    def parse_file(cls, path):
        token_stream = cls.read_tokens(path)

        def next_token():
            return next(token_stream)

        parsed = dict(objective_sense=1.0, variable_count=0, constraint_count=0,
                      variable_cones=[], constraint_cones=[], objective_terms={},
                      objective_constant=0.0, row_indices=[], column_indices=[],
                      values=[], constants={}, power_cones=[], power_star_cones=[],
                      matrix_sides=[], objective_matrix_terms=[], matrix_terms=[])

        while True:
            try:
                token = next_token()
            except StopIteration:
                break

            if token == "VER":
                next_token()

            elif token == "OBJSENSE":
                parsed["objective_sense"] = 1.0 if next_token().upper() == "MIN" else -1.0

            elif token in ("POWCONES", "POWSTARCONES"):
                key = "power_cones" if token == "POWCONES" else "power_star_cones"
                cone_count, _ = next_token().split()
                for _ in range(int(cone_count)):
                    exponent_count = int(next_token())
                    parsed[key].append(
                        [float(next_token()) for _ in range(exponent_count)])

            elif token in ("VAR", "CON"):
                total, block_count = next_token().split()
                parsed["variable_count" if token == "VAR" else "constraint_count"] = int(total)
                destination = parsed["variable_cones" if token == "VAR" else "constraint_cones"]
                for _ in range(int(block_count)):
                    cone_name, dimension = next_token().split()
                    destination.append((cone_name, int(dimension)))

            elif token == "INT":
                raise Unsupported("integer variables")

            elif token == "OBJACOORD":
                for _ in range(int(next_token())):
                    column, value = next_token().split()
                    parsed["objective_terms"][int(column)] = (
                        parsed["objective_terms"].get(int(column), 0.0) + float(value))

            elif token == "ACOORD":
                for _ in range(int(next_token())):
                    row, column, value = next_token().split()
                    parsed["row_indices"].append(int(row))
                    parsed["column_indices"].append(int(column))
                    parsed["values"].append(float(value))

            elif token == "BCOORD":
                for _ in range(int(next_token())):
                    row, value = next_token().split()
                    parsed["constants"][int(row)] = (
                        parsed["constants"].get(int(row), 0.0) + float(value))

            elif token == "OBJBCOORD":
                parsed["objective_constant"] = float(next_token())

            elif token == "PSDVAR":
                for _ in range(int(next_token())):
                    parsed["matrix_sides"].append(int(next_token()))

            elif token in ("OBJFCOORD", "FCOORD"):
                destination = (parsed["objective_matrix_terms"] if token == "OBJFCOORD"
                               else parsed["matrix_terms"])
                for _ in range(int(next_token())):
                    fields = next_token().split()
                    destination.append(
                        tuple(int(field) for field in fields[:-1]) + (float(fields[-1]),))

            elif token in ("PSDCON", "HCOORD", "DCOORD"):
                raise Unsupported("block %s" % token)

            else:
                raise Unsupported("unknown block %s" % token)

        return parsed

    @classmethod
    def emit_block(cls, kind, table_index, block_matrix, block_constant, parsed,
                   blocks, right_hand_sides, cones):
        dimension = block_matrix.shape[0]

        if kind == "F":
            return

        if kind == "L=":
            blocks.append(-block_matrix)
            right_hand_sides.append(block_constant)
            cones.append(("Zero", dimension, {}))

        elif kind == "L+":
            blocks.append(-block_matrix)
            right_hand_sides.append(block_constant)
            cones.append(("Nonneg", dimension, {}))

        elif kind == "L-":
            blocks.append(block_matrix)
            right_hand_sides.append(-block_constant)
            cones.append(("Nonneg", dimension, {}))

        elif kind in ("Q", "QR"):
            if kind == "QR":
                rotation = cls.rotation_for_rotated_cone(dimension)
                block_matrix = (rotation @ block_matrix).tocsc()
                block_constant = numpy.asarray(rotation @ block_constant).ravel()
            blocks.append(-block_matrix)
            right_hand_sides.append(block_constant)
            cones.append(("SecondOrder", dimension, {}))

        elif kind == "EXP":
            reversed_order = [2, 1, 0]
            blocks.append(-block_matrix[reversed_order])
            right_hand_sides.append(block_constant[reversed_order])
            cones.append(("Exponential", 3, {}))

        elif kind == "POW":
            blocks.append(-block_matrix)
            right_hand_sides.append(block_constant)
            cones.append(("Power", 3, {"alpha": cls.power_exponent(table_index, parsed)}))

        else:
            raise Unsupported("cone %s" % kind)

    @classmethod
    def load(cls, path):
        parsed = cls.parse_file(path)
        variable_count = parsed["variable_count"]
        constraint_count = parsed["constraint_count"]
        matrix_sides = parsed["matrix_sides"]

        matrix_variable_offsets = []
        total_variable_count = variable_count
        for side in matrix_sides:
            matrix_variable_offsets.append(total_variable_count)
            total_variable_count += cls.svec_length(side)
        root_two = math.sqrt(2.0)

        objective_gradient = numpy.zeros(total_variable_count)
        for column, value in parsed["objective_terms"].items():
            objective_gradient[column] = value
        for matrix, matrix_row, matrix_column, value in parsed["objective_matrix_terms"]:
            position = (matrix_variable_offsets[matrix]
                        + cls.svec_index(matrix_sides[matrix], matrix_row, matrix_column))
            objective_gradient[position] += (value if matrix_row == matrix_column
                                             else root_two * value)
        objective_gradient *= parsed["objective_sense"]

        row_indices = list(parsed["row_indices"])
        column_indices = list(parsed["column_indices"])
        values = list(parsed["values"])
        for row, matrix, matrix_row, matrix_column, value in parsed["matrix_terms"]:
            row_indices.append(row)
            column_indices.append(matrix_variable_offsets[matrix]
                                  + cls.svec_index(matrix_sides[matrix], matrix_row, matrix_column))
            values.append(value if matrix_row == matrix_column else root_two * value)

        if values:
            constraint_matrix = scipy.sparse.csc_matrix(
                (numpy.asarray(values, float),
                 (numpy.asarray(row_indices, numpy.int64),
                  numpy.asarray(column_indices, numpy.int64))),
                shape=(constraint_count, total_variable_count))
        else:
            constraint_matrix = scipy.sparse.csc_matrix(
                (constraint_count, total_variable_count))

        constraint_constants = numpy.zeros(constraint_count)
        for row, value in parsed["constants"].items():
            constraint_constants[row] = value

        blocks = []
        right_hand_sides = []
        cones = []

        offset = 0
        for cone_name, dimension in parsed["constraint_cones"]:
            kind, table_index = cls.split_cone_name(cone_name)
            cls.emit_block(kind, table_index,
                           constraint_matrix[offset:offset + dimension].tocsc(),
                           constraint_constants[offset:offset + dimension],
                           parsed, blocks, right_hand_sides, cones)
            offset += dimension

        identity = scipy.sparse.identity(total_variable_count, format="csc")
        offset = 0
        for cone_name, dimension in parsed["variable_cones"]:
            kind, table_index = cls.split_cone_name(cone_name)
            cls.emit_block(kind, table_index,
                           identity[:, offset:offset + dimension].T.tocsc(),
                           numpy.zeros(dimension),
                           parsed, blocks, right_hand_sides, cones)
            offset += dimension

        for matrix, side in enumerate(matrix_sides):
            dimension = cls.svec_length(side)
            start = matrix_variable_offsets[matrix]
            blocks.append(-identity[:, start:start + dimension].T.tocsc())
            right_hand_sides.append(numpy.zeros(dimension))
            cones.append(("PSDTriangle", side, {}))

        if not blocks:
            raise Unsupported("no constraints")

        return {
            "P": scipy.sparse.csc_matrix((total_variable_count, total_variable_count)),
            "q": objective_gradient,
            "E": scipy.sparse.vstack(blocks).tocsc(),
            "f": numpy.concatenate(right_hand_sides),
            "cones": cones,
            "objective_constant": parsed["objective_constant"] * parsed["objective_sense"],
        }
