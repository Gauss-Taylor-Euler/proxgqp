ZERO = 0
NONNEG = 1
SECOND_ORDER = 2
PSD_TRIANGLE = 3
EXPONENTIAL = 4
POWER = 5


class Cone:

    kind = None
    name = None

    def __init__(self, size, exponent=0.5):
        self.size = int(size)
        self.exponent = float(exponent)

    @property
    def rows(self):
        return self.size

    def __repr__(self):
        return "%s(%d)" % (self.name, self.size)

    def __eq__(self, other):
        return (isinstance(other, Cone) and self.kind == other.kind
                and self.size == other.size
                and self.exponent == other.exponent)


class Zero(Cone):

    kind = ZERO
    name = "Zero"


class Nonneg(Cone):

    kind = NONNEG
    name = "Nonneg"


class SecondOrder(Cone):

    kind = SECOND_ORDER
    name = "SecondOrder"

    def __init__(self, size):
        if int(size) < 2:
            raise ValueError("SecondOrder needs at least 2 rows, got %r" % (size,))
        Cone.__init__(self, size)


class PSDTriangle(Cone):

    kind = PSD_TRIANGLE
    name = "PSDTriangle"

    def __init__(self, side):
        if int(side) < 1:
            raise ValueError("PSDTriangle needs a side of at least 1, got %r" % (side,))
        Cone.__init__(self, side)

    @property
    def side(self):
        return self.size

    @property
    def rows(self):
        return self.size * (self.size + 1) // 2

    def __repr__(self):
        return "PSDTriangle(side=%d, rows=%d)" % (self.size, self.rows)


class Exponential(Cone):

    kind = EXPONENTIAL
    name = "Exponential"

    def __init__(self):
        Cone.__init__(self, 3)

    def __repr__(self):
        return "Exponential()"


class Power(Cone):

    kind = POWER
    name = "Power"

    def __init__(self, exponent):
        if not 0.0 < float(exponent) < 1.0:
            raise ValueError("Power needs an exponent in (0, 1), got %r" % (exponent,))
        Cone.__init__(self, 3, exponent)

    @property
    def alpha(self):
        return self.exponent

    def __repr__(self):
        return "Power(%.6g)" % self.exponent


def cone_rows(cones):
    return sum(cone.rows for cone in cones)
