"""Quadtree spatial index: splitting, merging, range queries and point lookup.

Every node covers a square region.  A leaf stores points, an inner node keeps
four children that cover its region in quarters.  Points on a split line belong
to the upper and the right side, so every point ends up in exactly one leaf.
"""

DEFAULT_CAPACITY = 4
DEFAULT_MAX_DEPTH = 6


class QuadtreeError(Exception):
    """Raised when a quadtree operation cannot be carried out."""


def _check_number(value, name):
    """Return value as a float, refusing everything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a number" % (name,))
    return float(value)


class Bounds(object):
    """An axis aligned rectangle; its borders count as inside."""

    __slots__ = ("x", "y", "width", "height")

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @classmethod
    def from_corners(cls, x0, y0, x1, y1):
        """The rectangle spanned by two opposite corners."""
        return cls(x0, y0, x1 - x0, y1 - y0)

    @property
    def x1(self):
        return self.x + self.width

    @property
    def y1(self):
        return self.y + self.height

    def contains(self, point):
        """Whether point is inside the rectangle, borders included."""
        return self.x <= point[0] <= self.x1 and self.y <= point[1] <= self.y1

    def intersects(self, other):
        """Whether the rectangle shares at least one point with other."""
        return not (
            other.x1 <= self.x
            or self.x1 <= other.x
            or other.y1 <= self.y
            or self.y1 <= other.y
        )

    def quadrant_of(self, point):
        """Index of the child quadrant that owns point."""
        right = point[0] >= self.x + self.width / 2.0
        upper = point[1] >= self.y + self.height / 2.0
        return (int(right) << 1) | int(upper)

    def child_bounds(self, index):
        """The region of the child quadrant at index."""
        half_x = self.width / 2.0
        half_y = self.height / 2.0
        x = self.x + (half_x if index & 1 else 0.0)
        y = self.y + (half_y if index & 2 else 0.0)
        return Bounds(x, y, half_x, half_y)

    def __repr__(self):
        return "Bounds(%g, %g, %g, %g)" % (self.x, self.y, self.width, self.height)


class Node(object):
    """One node of the quadtree."""

    __slots__ = ("bounds", "depth", "points", "children")

    def __init__(self, bounds, depth):
        self.bounds = bounds
        self.depth = depth
        self.points = []
        self.children = None

    @property
    def is_leaf(self):
        return self.children is None

    def __repr__(self):
        return "Node(depth=%d, points=%d)" % (self.depth, len(self.points))


class Quadtree(object):
    """A quadtree over one square region, standard library only."""

    def __init__(self, x, y, size, capacity=DEFAULT_CAPACITY,
                 max_depth=DEFAULT_MAX_DEPTH):
        x = _check_number(x, "x")
        y = _check_number(y, "y")
        size = _check_number(size, "size")
        if size <= 0.0:
            raise ValueError("size must be positive")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an int")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if isinstance(max_depth, bool) or not isinstance(max_depth, int):
            raise TypeError("max_depth must be an int")
        if max_depth < 0:
            raise ValueError("max_depth must not be negative")
        self.capacity = capacity
        self.max_depth_limit = max_depth
        self.root = Node(Bounds(x, y, size, size), 0)
        self._count = 0

    def __len__(self):
        return self._count

    def __repr__(self):
        return "Quadtree(%r, points=%d)" % (self.root.bounds, len(self))

    # -- public API ----------------------------------------------------
    def insert(self, point):
        """Add one point; the same coordinate may be added several times."""
        point = self._check_point(point)
        self._insert(self.root, point)
        self._count += 1

    def insert_many(self, points):
        """Add every point of the iterable."""
        for point in points:
            self.insert(point)

    def remove(self, point):
        """Remove one copy of point; False when no copy is stored."""
        point = self._check_point(point)
        path = self._descend(point)
        leaf = path[-1]
        if point not in leaf.points:
            return False
        leaf.points.remove(point)
        self._count -= 1
        self._collapse(path[:-1])
        return True

    def count(self, point):
        """How many copies of point are stored."""
        leaf = self.locate(point)
        return len(leaf.points)

    def locate(self, point):
        """The leaf node that owns point."""
        point = self._check_point(point)
        return self._descend(point)[-1]

    def query(self, x0, y0, x1, y1):
        """Every stored point inside the rectangle; its borders count."""
        rect = _check_rect(x0, y0, x1, y1)
        found = []
        self._collect(self.root, rect, found)
        return found

    def iter_leaves(self):
        """The leaves of the tree, depth first."""
        return self._walk(self.root)

    def max_depth(self):
        """Depth of the deepest leaf."""
        depth = 0
        for leaf in self.iter_leaves():
            if leaf.depth > depth:
                depth = leaf.depth
        return depth

    def node_count(self):
        """Number of nodes in the tree."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            total += 1
            if node.children is not None:
                stack.extend(node.children)
        return total

    # -- internals -----------------------------------------------------
    def _check_point(self, point):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise TypeError("point must be a pair of coordinates")
        x = _check_number(point[0], "point x")
        y = _check_number(point[1], "point y")
        if not self.root.bounds.contains((x, y)):
            raise ValueError("point outside the tree: %r" % (point,))
        return (x, y)

    def _insert(self, node, point):
        while node.children is not None:
            node = node.children[node.bounds.quadrant_of(point)]
        if point not in node.points:
            node.points.append(point)
        if len(node.points) >= self.capacity and node.depth <= self.max_depth_limit:
            self._split(node)

    def _split(self, node):
        """Turn a leaf into an inner node and hand its points to the children."""
        node.children = [
            Node(node.bounds.child_bounds(index), node.depth + 1)
            for index in range(4)
        ]
        points = node.points
        node.points = []
        for point in points:
            child = node.children[node.bounds.quadrant_of(point)]
            child.points.append(point)

    def _descend(self, point):
        """The chain of nodes from the root down to the leaf that owns point."""
        node = self.root
        path = [node]
        while node.children is not None:
            node = node.children[node.bounds.quadrant_of(point)]
            path.append(node)
        return path

    def _collect(self, node, rect, found):
        if not node.bounds.intersects(rect):
            return
        for point in node.points:
            if rect.contains(point):
                found.append(point)
        for child in node.children or ():
            self._collect(child, rect, found)

    def _collapse(self, parents):
        """Merge the children of the point that was removed back into it."""
        for node in parents:
            if any(child.children is not None for child in node.children):
                continue
            total = 0
            for child in node.children:
                total += len(child.points)
            if total > self.capacity:
                continue
            node.points = [point for child in node.children for point in child.points]
            node.children = None

    def _walk(self, node):
        if node.children is None:
            yield node
            return
        for child in node.children:
            for leaf in self._walk(child):
                yield leaf


def _check_rect(x0, y0, x1, y1):
    """Build the query rectangle from two opposite corners."""
    x0 = _check_number(x0, "x0")
    y0 = _check_number(y0, "y0")
    x1 = _check_number(x1, "x1")
    y1 = _check_number(y1, "y1")
    if x1 < x0 or y1 < y0:
        raise ValueError("rectangle corners are inverted")
    return Bounds.from_corners(x0, y0, x1, y1)
