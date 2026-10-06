"""Tests for the quadtree kernel (unittest, standard library only)."""

import unittest

from quadtree.core import Quadtree


class QueryTests(unittest.TestCase):
    def test_query_returns_the_points_inside_the_rectangle(self):
        tree = Quadtree(0, 0, 8, capacity=2, max_depth=4)
        tree.insert_many([(4, 2), (6, 2), (6, 6), (2, 6)])
        self.assertEqual(sorted(tree.query(5, 1, 7, 3)), [(6, 2)])
        self.assertEqual(sorted(tree.query(1, 5, 3, 7)), [(2, 6)])
        self.assertEqual(sorted(tree.query(1, 0, 4, 4)), [(4, 2)])
        self.assertEqual(sorted(tree.query(4, 0, 4, 4)), [(4, 2)])
        self.assertEqual(
            sorted(tree.query(0, 0, 8, 8)), [(2, 6), (4, 2), (6, 2), (6, 6)]
        )
        self.assertEqual(tree.query(0, 0, 3, 3), [])


class LocateTests(unittest.TestCase):
    def test_locate_returns_the_leaf_that_owns_the_point(self):
        tree = Quadtree(0, 0, 16, capacity=2, max_depth=4)
        points = [(1, 1), (1, 9), (9, 1), (9, 9), (13, 3), (3, 13), (7, 7)]
        tree.insert_many(points)
        for point in points:
            leaf = tree.locate(point)
            self.assertTrue(leaf.is_leaf, "not a leaf for %r" % (point,))
            self.assertTrue(
                leaf.bounds.contains(point),
                "%r is not inside %r" % (point, leaf.bounds),
            )
            self.assertIn(point, leaf.points)
        boundary = Quadtree(0, 0, 8, capacity=1, max_depth=4)
        boundary.insert((4, 2))
        boundary.insert((3, 2))
        self.assertTrue(boundary.locate((4, 2)).bounds.contains((4, 2)))
        self.assertNotEqual(
            boundary.locate((4, 2)).bounds.x, boundary.locate((3, 2)).bounds.x
        )


class SplitTests(unittest.TestCase):
    def test_splitting_keeps_every_inserted_copy(self):
        tree = Quadtree(0, 0, 16, capacity=2, max_depth=3)
        points = [(8, 8), (8, 8), (8, 8), (7, 7), (12, 4), (4, 12)]
        tree.insert_many(points)
        self.assertEqual(len(tree), len(points))
        self.assertEqual(sorted(tree.query(0, 0, 16, 16)), sorted(points))
        stored = []
        for leaf in tree.iter_leaves():
            stored.extend(leaf.points)
        self.assertEqual(sorted(stored), sorted(points))

    def test_clustered_points_reach_the_depth_limit(self):
        tree = Quadtree(0, 0, 16, capacity=2, max_depth=3)
        cluster = [(2.1, 2.1), (2.2, 2.2), (2.3, 2.1)]
        tree.insert_many(cluster)
        self.assertEqual(tree.max_depth(), 3)
        deepest = [
            leaf for leaf in tree.iter_leaves() if leaf.depth == 3 and leaf.points
        ]
        self.assertEqual(len(deepest), 1)
        self.assertEqual(sorted(deepest[0].points), sorted(cluster))
        for leaf in tree.iter_leaves():
            if leaf.depth < 3:
                self.assertLessEqual(len(leaf.points), 2)

    def test_a_leaf_splits_only_after_capacity_is_exceeded(self):
        tree = Quadtree(0, 0, 16, capacity=4, max_depth=5)
        tree.insert_many([(2, 2), (2, 13), (13, 2), (13, 13)])
        self.assertEqual(len(list(tree.iter_leaves())), 1)
        self.assertEqual(tree.max_depth(), 0)
        self.assertEqual(tree.node_count(), 1)
        tree.insert((13, 3))
        self.assertEqual(len(list(tree.iter_leaves())), 4)
        self.assertEqual(tree.max_depth(), 1)
        self.assertEqual(tree.node_count(), 5)


class RemoveTests(unittest.TestCase):
    def test_removing_points_merges_the_tree_back(self):
        tree = Quadtree(0, 0, 16, capacity=2, max_depth=4)
        tree.insert_many([(2.1, 2.1), (2.2, 2.2), (2.3, 2.1)])
        self.assertGreater(len(list(tree.iter_leaves())), 1)
        self.assertTrue(tree.remove((2.2, 2.2)))
        self.assertFalse(tree.remove((5.0, 5.0)))
        self.assertEqual(len(tree), 2)
        self.assertEqual(len(list(tree.iter_leaves())), 1)
        self.assertEqual(tree.max_depth(), 0)
        self.assertEqual(sorted(tree.query(0, 0, 16, 16)), [(2.1, 2.1), (2.3, 2.1)])
        self.assertTrue(tree.remove((2.1, 2.1)))
        self.assertTrue(tree.remove((2.3, 2.1)))
        self.assertEqual(len(tree), 0)
        self.assertEqual(len(list(tree.iter_leaves())), 1)
        self.assertEqual(tree.max_depth(), 0)


class CountTests(unittest.TestCase):
    def test_duplicate_copies_are_counted_one_by_one(self):
        tree = Quadtree(0, 0, 16, capacity=2, max_depth=4)
        for _ in range(5):
            tree.insert((7, 7))
        self.assertEqual(len(tree), 5)
        self.assertEqual(tree.count((7, 7)), 5)
        self.assertEqual(sorted(tree.query(0, 0, 16, 16)), [(7, 7)] * 5)
        self.assertEqual(tree.count((3, 3)), 0)
        self.assertTrue(tree.remove((7, 7)))
        self.assertEqual(tree.count((7, 7)), 4)

        loose = Quadtree(0, 0, 16, capacity=8)
        loose.insert_many([(1, 1), (1, 9), (9, 1), (9, 9), (1, 1), (1, 1)])
        self.assertEqual(loose.count((1, 1)), 3)
        self.assertEqual(loose.count((1, 9)), 1)
        self.assertEqual(loose.count((4, 4)), 0)


class OwnershipTests(unittest.TestCase):
    def test_every_leaf_owns_only_points_inside_its_region(self):
        tree = Quadtree(-4, -4, 8, capacity=2, max_depth=3)
        points = []
        for index in range(24):
            points.append(
                (float((index * 5) % 8) - 4.0, float((index * 3) % 8) - 4.0)
            )
        tree.insert_many(points)
        self.assertEqual(len(tree), len(points))
        owned = 0
        for leaf in tree.iter_leaves():
            for point in leaf.points:
                self.assertTrue(
                    leaf.bounds.contains(point),
                    "%r is not inside %r" % (point, leaf.bounds),
                )
                self.assertTrue(tree.root.bounds.contains(point))
                owned += 1
        self.assertEqual(owned, len(points))
        self.assertEqual(len(tree), owned)


class InputTests(unittest.TestCase):
    def test_invalid_calls_are_rejected(self):
        tree = Quadtree(0, 0, 8, capacity=2, max_depth=2)
        with self.assertRaises(ValueError):
            Quadtree(0, 0, 0)
        with self.assertRaises(ValueError):
            Quadtree(0, 0, 8, capacity=0)
        with self.assertRaises(ValueError):
            Quadtree(0, 0, 8, max_depth=-1)
        with self.assertRaises(ValueError):
            tree.insert((9, 1))
        with self.assertRaises(ValueError):
            tree.insert((1, -0.5))
        with self.assertRaises(ValueError):
            tree.query(4, 4, 1, 1)
        with self.assertRaises(ValueError):
            tree.locate((8, 9))
        with self.assertRaises(TypeError):
            tree.insert(3)
        with self.assertRaises(TypeError):
            tree.query(0, 0, 8, "8")
        self.assertEqual(tree.count((4, 4)), 0)
        self.assertFalse(tree.remove((4, 4)))
        self.assertTrue(tree.root.is_leaf)


if __name__ == "__main__":
    unittest.main()
