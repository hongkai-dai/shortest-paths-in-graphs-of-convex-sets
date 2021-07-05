import numpy as np
import pydrake.solvers.mathematicalprogram as mp


class ShortestPathVariables():
    def __init__(self, phi: np.ndarray, l: np.ndarray, x: np.ndarray):
        self.phi = phi
        self.l = l
        self.x = x

    @staticmethod
    def populate_program(prog, graph, relaxation=True):
        phi_type = prog.NewContinuousVariables if relaxation else\
            prog.NewBinaryVariables
        phi = phi_type(graph.n_edges)
        if relaxation:
            prog.AddBoundingBoxConstraint(0, 1, phi)
        l = prog.NewContinuousVariables(graph.n_edges)
        x = prog.NewContinuousVariables(graph.n_sets, graph.dimension)
        return ShortestPathVariables(phi, l, x)


class ShortestPathConstraints():

    def __init__(self, cons, deg, sp_cons, obj=None):
        # not all constraints of the spp are stored here
        # only the ones we care of (the linear ones)
        self.conservation = cons
        self.degree = deg
        self.spatial_conservation = sp_cons
        self.objective = obj

    @staticmethod
    def populate_program(prog, graph, vars):

        # containers for the constraints we want to keep track of
        cons = []
        deg = []
        sp_cons = []

        for vertex, graph_set in graph.sets.items():
            edges_in = graph.incoming_edges(vertex)[1]
            edges_out = graph.outgoing_edges(vertex)[1]

            phi_in = sum(vars.phi[edges_in])
            phi_out = sum(vars.phi[edges_out])

            delta_sv = 1 if vertex == graph.source else 0
            delta_tv = 1 if vertex == graph.target else 0

            # conservation of flow
            if len(edges_in) > 0 or len(edges_out) > 0:
                residual = phi_out + delta_tv - phi_in - delta_sv
                cons.append(prog.AddLinearConstraint(residual == 0))

            # degree constraints
            if len(edges_out) > 0:
                residual = phi_out + delta_tv - 1
                deg.append(prog.AddLinearConstraint(residual <= 0))

            # spatial constraint x ∈ graph_set
            graph_set.add_membership_constraint(
                prog, vars.x[graph.vertex_index(vertex)])

        for k, edge in enumerate(graph.edges):
            graph.lengths[edge].add_constraint_nonlinear(
                prog, vars.l[k],
                np.concatenate((vars.x[graph.vertex_index(edge[0])],
                                vars.x[graph.vertex_index(edge[1])])))

        return ShortestPathConstraints(cons, deg, sp_cons)


class ShortestPathProblem():
    def __init__(self, graph, relaxation=True):
        self.graph = graph
        self.relaxation = relaxation

        self.prog = mp.MathematicalProgram()
        self.vars = ShortestPathVariables.populate_program(
            self.prog, graph, relaxation)
        self.constraints = ShortestPathConstraints.populate_program(
            self.prog, graph, self.vars)
        self.prog.AddQuadraticCost(
            sum(self.vars.l * self.vars.phi), is_convex=False)
