import numpy as np

from utils.convex_sets import Singleton, Polyhedron, Ellipsoid
from utils.convex_functions import TwoNorm, SquaredTwoNorm
from utils.graph import GraphOfConvexSets
from utils.shortest_path_bilinear import ShortestPathProblem

import unittest

from pydrake.solvers.snopt import SnoptSolver

import matplotlib.pyplot as plt


class TestSingletons(unittest.TestCase):
    def test_unique_path(self):
        # There is a unique shortest path
        singletons = (
            Singleton((0, 0)),
            Singleton((1, 1)),
            Singleton((1, 0)),
            Singleton((2, 1)),
            Singleton((3, 3)),
        )

        vertices = ['s', 'v1', 'v2', 'v3', 't']

        G = GraphOfConvexSets()
        G.add_sets(singletons, vertices)
        G.set_source('s')
        G.set_target('t')

        H = np.hstack((np.eye(2), -np.eye(2)))
        l = TwoNorm(H)
        edges = {
            "s": ("v1", "v2"),
            "v1": ("v3", "t"),
            "v2": ("v3", ),
            "v3": ("t", ),
        }
        for u, vs in edges.items():
            for v in vs:
                G.add_edge(u, v, l)

        plt.figure()
        G.draw_sets()
        G.draw_edges()
        G.label_sets()

        spp = ShortestPathProblem(G, relaxation=True)
        snopt_solver = SnoptSolver()
        result = snopt_solver.Solve(spp.prog)
        self.assertTrue(result.is_success())
        phi_sol = result.GetSolution(spp.vars.phi)
        for edge_index, edge in enumerate(spp.graph.edges):
            if edge in (('s', 'v1'), ('v1', 't')):
                self.assertAlmostEqual(phi_sol[edge_index], 1)
            else:
                self.assertAlmostEqual(phi_sol[edge_index], 0)

    def test_nonunique_path(self):
        # There are multiple shortest paths
        singletons = (
            Singleton((0, 0)),
            Singleton((1, 1)),
            Singleton((1, 0)),
            Singleton((0, 1)),
            Singleton((2, 2)),
            Singleton((3, 3)),
        )

        vertices = ['s', 'v1', 'v2', 'v3', 'v4', 't']

        G = GraphOfConvexSets()
        G.add_sets(singletons, vertices)
        G.set_source('s')
        G.set_target('t')

        H = np.hstack((np.eye(2), -np.eye(2)))
        l = TwoNorm(H)
        edges = {
            "s": ("v1", "v2", "v3"),
            "v1": ("v4", ),
            "v2": ("v1", "t"),
            "v3": (
                "v1",
                "t",
            ),
        }
        for u, vs in edges.items():
            for v in vs:
                G.add_edge(u, v, l)

        plt.figure()
        G.draw_sets()
        G.draw_edges()
        G.label_sets()

        spp = ShortestPathProblem(G, relaxation=True)
        snopt_solver = SnoptSolver()
        result = snopt_solver.Solve(spp.prog)
        self.assertTrue(result.is_success())
        phi_sol = result.GetSolution(spp.vars.phi)
        path1 = (('s', 'v2'), ('v2', 't'))
        path2 = (('s', 'v3'), ('v3', 't'))
        phi_sol_expected1 = np.zeros((len(spp.graph.edges), ))
        phi_sol_expected2 = np.zeros((len(spp.graph.edges), ))
        for edge in path1:
            phi_sol_expected1[spp.graph.edge_index(edge)] = 1
        for edge in path2:
            phi_sol_expected2[spp.graph.edge_index(edge)] = 1
        self.assertTrue(
            np.linalg.norm(phi_sol - phi_sol_expected1) < 1E-5
            or np.linalg.norm(phi_sol - phi_sol_expected2) < 1E-5)


if __name__ == "__main__":
    unittest.main()
