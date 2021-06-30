import numpy as np

class ConvexFunction():
    """Parent class for all the convex functions."""

    def add_perspective_constraint(self, prog, slack, scale, x):

        cost = self._add_perspective_constraint(prog, slack, scale, x)
        domain = self.enforce_domain(prog, scale, x)

        return cost, domain

    def check_perspective_constraint(self, slack, scale, x, tol):
        self._check_perspective_constraint(slack, scale, x, tol)

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

    def _check_perspective_constraint(self, slack, scale, x, tol):
        assert slack + tol >= self.c * scale, f"slack {slack}, c * scale {self.c * scale}"

class TwoNorm(ConvexFunction):
    """Function of the form ||H x||_2 for x in D, where D is a ConvexSet."""

    def __init__(self, H, D=None):

        self.H = H
        self.D = D

    def _add_perspective_constraint(self, prog, slack, scale, x):

        Hx = self.H.dot(x)
        return prog.AddLorentzConeConstraint(slack, Hx.dot(Hx))

    def _check_perspective_constraint(self, slack, scale, x, tol):
        Hx = self.H.dot(x)
        assert slack + tol >= np.linalg.norm(Hx), f"slack {slack}, Hx norm {np.linalg.norm(Hx)}"

class SquaredTwoNorm(ConvexFunction):
    """Function of the form ||H x||_2^2 for x in D, where D is a ConvexSet."""

    def __init__(self, H, D=None):

        self.H = H
        self.D = D

    def _add_perspective_constraint(self, prog, slack, scale, x):
        Hx = self.H.dot(x)
        return prog.AddRotatedLorentzConeConstraint(slack, scale, Hx.dot(Hx))

    def _check_perspective_constraint(self, slack, scale, x, tol):
        assert slack >= -tol, f"{slack}"
        assert scale >= -tol, f"{scale}"
        Hx = self.H.dot(x)
        assert slack * scale + tol >= np.sum(Hx ** 2), f"slack * scale {slack * scale}, Hx squared norm {np.sum(Hx ** 2)}"
