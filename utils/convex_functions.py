import numpy as np
import pydrake.solvers.mathematicalprogram as mp

class ConvexFunction():
    """Parent class for all the convex functions."""

    def add_perspective_constraint(self, prog, slack, scale, x):

        cost = self._add_perspective_constraint(prog, slack, scale, x)
        domain = self.enforce_domain(prog, scale, x)

        return cost, domain

    def add_constraint_nonlinear(self, prog, slack, x):
        constraint = self._add_constraint_nonlinear(prog, slack, x)
        domain = self.enforce_domain(prog, 1., x)
        return constraint, domain

    def enforce_domain(self, prog, scale, x):
        if self.D is not None:
            return self.D.add_perspective_constraint(prog, scale, x)

class Constant(ConvexFunction):
    """Function of the form c for x in D, where D is a ConvexSet."""

    def __init__(self, c, D=None):

        self.c = c
        self.D = D

    def _add_perspective_constraint(self, prog, slack, scale, x):

        return prog.AddLinearConstraint(slack >= self.c * scale)

    def _add_constraint_nonlinear(self, prog, slack, x):
        return [prog.AddLinearConstraint(slack = self.c)]


class TwoNorm(ConvexFunction):
    """Function of the form ||H x||_2 for x in D, where D is a ConvexSet."""

    def __init__(self, H, D=None):

        self.H = H
        self.D = D

    def _add_perspective_constraint(self, prog, slack, scale, x):

        Hx = self.H.dot(x)
        return prog.AddLorentzConeConstraint(slack, Hx.dot(Hx))

    def _add_constraint_nonlinear(self, prog, slack, x):
        lorentz_A = np.zeros((1 + self.H.shape[0], 1 + self.H.shape[1]))
        lorentz_A[0, 0] = 1
        lorentz_A[1:, 1:] = self.H
        lorentz_b = np.zeros((1 + self.H.shape[0],))
        lorentz_cone_constraint = mp.LorentzConeConstraint(
            lorentz_A, lorentz_b,
            mp.LorentzConeConstraint.EvalType.kConvexSmooth)
        constraint = prog.AddConstraint(
            lorentz_cone_constraint, np.concatenate(([slack], x)))
        lorentz_cone_constraint.set_description(
            "two norm lorentz cone constraint")
        return [constraint]


class SquaredTwoNorm(ConvexFunction):
    """Function of the form ||H x||_2^2 for x in D, where D is a ConvexSet."""

    def __init__(self, H, D=None):

        self.H = H
        self.D = D

    def _add_perspective_constraint(self, prog, slack, scale, x):

        Hx = self.H.dot(x)
        return prog.AddRotatedLorentzConeConstraint(slack, scale, Hx.dot(Hx))

    def _add_constraint_nonlinear(self, prog, slack, x):
        """ Add constraint slack = |H*x|² """
        Hx = self.H.dot(x)
        # The constraint is [    x]ᵀ * [HᵀH 0] * [    x] - [0 1]*[    x]  = 0
        #                   [slack]    [0   0]   [slack]         [slack]
        x_dim = x.shape[0]
        Hessian = np.zeros((x_dim+1, x_dim+1))
        Hessian[:x_dim, :x_dim] = 2 * self.H.T.dot(self.H)
        linear_coeff = np.zeros(x_dim + 1)
        linear_coeff[-1] = -1
        constraint = prog.AddConstraint(mp.QuadraticConstraint(
            Hessian, linear_coeff, 0., 0.), np.concatenate((x, [slack])))
        constraint.evaluator().set_description(
            "squared two-norm quadratic constraint")
        return [constraint]
