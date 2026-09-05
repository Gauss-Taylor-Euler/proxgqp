import numpy
import scipy.sparse as sparse

import proxgqp

identity = sparse.csc_matrix(numpy.eye(3))
bound = numpy.array([1.0, 0.0, 0.0])
problem = proxgqp.Problem(identity, numpy.array([-1.0, -1.0, -1.0]),
                          identity, bound, [proxgqp.SecondOrder(3)])
solution = proxgqp.solve(problem)
assert solution.status == "solved", solution.status

slack = bound - solution.x
assert slack[0] >= numpy.linalg.norm(slack[1:]) - 1e-9, slack
assert abs(slack[0] - numpy.linalg.norm(slack[1:])) < 1e-6, slack
