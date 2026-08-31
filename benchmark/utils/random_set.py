import numpy

from generate import build
from identifier import decode, encode


class RandomSet:

    name = None
    configurations = ()
    rows_per_variable = 10

    @classmethod
    def list_problems(cls):
        return [encode(fields) for fields in cls.configurations]

    @classmethod
    def variable_count(cls, parameters):
        return parameters["n"]

    @classmethod
    def cone_specification(cls, parameters):
        raise NotImplementedError

    @classmethod
    def load(cls, identifier):
        parameters = decode(identifier)
        generator = numpy.random.default_rng(parameters["s"])
        variable_count = cls.variable_count(parameters)
        problem, _solution, _dual = build(
            variable_count,
            cls.cone_specification(parameters),
            generator,
            per_row=min(cls.rows_per_variable, variable_count),
            objective_rank=cls.objective_rank(parameters, variable_count),
            condition_number=10.0 ** parameters.get("c", 3))
        return problem

    @classmethod
    def objective_rank(cls, parameters, variable_count):
        requested = parameters.get("r", -1)
        return variable_count if requested < 0 else requested


