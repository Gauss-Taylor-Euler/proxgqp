import math

import numpy


def block_rows(kind, size):
    if kind in ("Exponential", "Power"):
        return 3
    if kind == "PSDTriangle":
        return size * (size + 1) // 2
    return size


def project_nonneg(vector):
    return numpy.maximum(vector, 0.0)


def project_second_order(vector):
    apex = vector[0]
    tail = vector[1:]
    tail_norm = float(numpy.linalg.norm(tail))
    if tail_norm <= apex:
        return vector.copy()
    if tail_norm <= -apex:
        return numpy.zeros_like(vector)
    scale = 0.5 * (1.0 + apex / tail_norm)
    return numpy.concatenate([[scale * tail_norm], scale * tail])


def triangle_side(row_count):
    side = int(round((math.sqrt(8 * row_count + 1) - 1) / 2))
    if side * (side + 1) // 2 != row_count:
        raise ValueError("%d is not a triangular number" % row_count)
    return side


def svec_to_matrix(vector):
    side = triangle_side(vector.size)
    matrix = numpy.zeros((side, side))
    root_two = math.sqrt(2.0)
    position = 0
    for column in range(side):
        for row in range(column, side):
            value = vector[position] if row == column else vector[position] / root_two
            matrix[row, column] = value
            matrix[column, row] = value
            position += 1
    return matrix


def matrix_to_svec(matrix):
    side = matrix.shape[0]
    vector = numpy.empty(side * (side + 1) // 2)
    root_two = math.sqrt(2.0)
    position = 0
    for column in range(side):
        for row in range(column, side):
            vector[position] = (matrix[row, column] if row == column
                                else root_two * matrix[row, column])
            position += 1
    return vector


def project_psd(vector):
    matrix = svec_to_matrix(vector)
    eigenvalues, eigenvectors = numpy.linalg.eigh(matrix)
    clipped = (eigenvectors * numpy.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return matrix_to_svec(clipped)


def inside_exponential(point, tolerance=1e-9):
    x, y, z = point
    scale = max(1.0, abs(x), abs(y), abs(z))
    if y > tolerance * scale:
        return z > 0.0 and math.log(y) + x / y <= math.log(z) + tolerance
    return abs(y) <= tolerance * scale and x <= tolerance * scale and z >= -tolerance * scale


def inside_exponential_dual(point, tolerance=1e-9):
    u, w, t = point
    scale = max(1.0, abs(u), abs(w), abs(t))
    if u < -tolerance * scale:
        return t > 0.0 and math.log(-u) + w / u <= math.log(t) + 1.0 + tolerance
    return abs(u) <= tolerance * scale and w >= -tolerance * scale and t >= -tolerance * scale


def exponential_boundary_gap(slope, point):
    x0, y0, z0 = point
    growth = math.exp(slope)
    from_first = (x0 + growth * z0) * (1.0 + (1.0 - slope) * growth * growth)
    from_second = (y0 + (1.0 - slope) * growth * z0) * (slope + growth * growth)
    return from_second - from_first


def exponential_boundary_point(slope, point):
    x0, y0, z0 = point
    growth = math.exp(slope)
    denominator = 1.0 + (1.0 - slope) * growth * growth
    if abs(denominator) < 1e-300:
        return None
    height = (y0 + (1.0 - slope) * growth * z0) / denominator
    if height <= 0.0:
        return None
    return numpy.array([slope * height, height, height * growth])


def project_exponential(point):
    point = numpy.asarray(point, float)
    if inside_exponential(point):
        return point.copy()
    if inside_exponential_dual(-point):
        return numpy.zeros(3)

    x0, y0, z0 = point
    best = numpy.zeros(3)
    best_distance = float(numpy.dot(point, point))

    def consider(candidate):
        nonlocal best, best_distance
        if candidate is None or not numpy.all(numpy.isfinite(candidate)):
            return
        if not inside_exponential(candidate, 1e-8):
            return
        offset = candidate - point
        distance = float(numpy.dot(offset, offset))
        if distance < best_distance:
            best_distance = distance
            best = candidate

    consider(numpy.array([min(x0, 0.0), 0.0, max(z0, 0.0)]))
    if y0 > 0.0:
        ratio = x0 / y0
        if ratio < 100.0:
            floor = y0 * math.exp(ratio)
            if math.isfinite(floor) and abs(floor) < 1e100:
                consider(numpy.array([x0, y0, max(z0, floor)]))

    lowest, highest, samples = -60.0, 60.0, 1500
    previous_slope = lowest
    previous_gap = exponential_boundary_gap(lowest, point)
    for step in range(1, samples):
        slope = lowest + (highest - lowest) * step / (samples - 1)
        gap = exponential_boundary_gap(slope, point)
        bracketed = (math.isfinite(previous_gap) and math.isfinite(gap)
                     and previous_gap != 0.0 and previous_gap * gap < 0.0)
        if bracketed:
            left, right, left_gap = previous_slope, slope, previous_gap
            for _ in range(200):
                middle = 0.5 * (left + right)
                middle_gap = exponential_boundary_gap(middle, point)
                if not math.isfinite(middle_gap):
                    break
                if left_gap * middle_gap <= 0.0:
                    right = middle
                else:
                    left, left_gap = middle, middle_gap
                if right - left < 1e-14 * max(1.0, abs(left)):
                    break
            consider(exponential_boundary_point(0.5 * (left + right), point))
        previous_slope, previous_gap = slope, gap
    return best


def inside_power(point, exponent, tolerance=1e-9):
    x, y, z = point
    scale = max(1.0, abs(x), abs(y), abs(z))
    if x < -tolerance * scale or y < -tolerance * scale:
        return False
    x, y = max(x, 0.0), max(y, 0.0)
    if x == 0.0 or y == 0.0:
        return abs(z) <= tolerance * scale
    if z == 0.0:
        return True
    return (exponent * math.log(x) + (1.0 - exponent) * math.log(y)
            >= math.log(abs(z)) - tolerance)


def inside_power_dual(point, exponent, tolerance=1e-9):
    rescaled = numpy.array([point[0] / exponent, point[1] / (1.0 - exponent), point[2]])
    return inside_power(rescaled, exponent, tolerance)


def positive_quadratic_root(linear, constant):
    if constant <= 0.0:
        return max(linear, 0.0)
    discriminant = math.sqrt(linear * linear + 4.0 * constant)
    if linear >= 0.0:
        return 0.5 * (linear + discriminant)
    return 2.0 * constant / (discriminant - linear)


def power_boundary_point(multiplier, point, exponent):
    x0, y0, z0 = point
    height = abs(z0) - multiplier
    x = positive_quadratic_root(x0, multiplier * exponent * height)
    y = positive_quadratic_root(y0, multiplier * (1.0 - exponent) * height)
    return x, y, height


def power_boundary_gap(multiplier, point, exponent):
    x, y, height = power_boundary_point(multiplier, point, exponent)
    if x <= 0.0 or y <= 0.0:
        return -height
    return math.exp(exponent * math.log(x)
                    + (1.0 - exponent) * math.log(y)) - height


def project_power(point, exponent):
    point = numpy.asarray(point, float)
    if inside_power(point, exponent):
        return point.copy()
    if inside_power_dual(-point, exponent):
        return numpy.zeros(3)

    x0, y0, z0 = point
    best = numpy.zeros(3)
    best_distance = float(numpy.dot(point, point))

    def consider(candidate):
        nonlocal best, best_distance
        if not numpy.all(numpy.isfinite(candidate)):
            return
        if not inside_power(candidate, exponent, 1e-8):
            return
        offset = candidate - point
        distance = float(numpy.dot(offset, offset))
        if distance < best_distance:
            best_distance = distance
            best = candidate

    consider(numpy.array([0.0, max(y0, 0.0), 0.0]))
    consider(numpy.array([max(x0, 0.0), 0.0, 0.0]))
    consider(numpy.array([max(x0, 0.0), max(y0, 0.0), 0.0]))

    sign = 1.0 if z0 >= 0.0 else -1.0
    magnitude = abs(z0)
    if magnitude > 0.0:
        left, right = 0.0, magnitude
        left_gap = power_boundary_gap(left, point, exponent)
        right_gap = power_boundary_gap(right, point, exponent)
        if (math.isfinite(left_gap) and math.isfinite(right_gap)
                and (left_gap < 0.0) != (right_gap < 0.0)):
            for _ in range(400):
                middle = 0.5 * (left + right)
                middle_gap = power_boundary_gap(middle, point, exponent)
                if (left_gap < 0.0) != (middle_gap < 0.0):
                    right = middle
                else:
                    left, left_gap = middle, middle_gap
                if right - left <= 1e-15 * max(right, 1e-300):
                    break
            x, y, height = power_boundary_point(0.5 * (left + right), point, exponent)
            consider(numpy.array([max(x, 0.0), max(y, 0.0), sign * max(height, 0.0)]))
    return best


def project_dual(cones, vector):
    vector = numpy.asarray(vector, float)
    projected = numpy.empty_like(vector)
    offset = 0
    for kind, size, params in cones:
        rows = block_rows(kind, size)
        block = vector[offset:offset + rows]
        if kind == "Zero":
            projected[offset:offset + rows] = block
        elif kind == "Nonneg":
            projected[offset:offset + rows] = project_nonneg(block)
        elif kind == "SecondOrder":
            projected[offset:offset + rows] = project_second_order(block)
        elif kind == "PSDTriangle":
            projected[offset:offset + rows] = project_psd(block)
        elif kind == "Exponential":
            projected[offset:offset + rows] = block + project_exponential(-block)
        elif kind == "Power":
            projected[offset:offset + rows] = block + project_power(-block, params["alpha"])
        else:
            raise ValueError("unknown cone kind %r" % kind)
        offset += rows
    return projected
