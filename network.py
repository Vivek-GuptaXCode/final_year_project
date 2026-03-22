import networkx as nx
import math

class RoadNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.positions = {}

    def add_node(self, node, pos):
        self.graph.add_node(node)
        self.positions[node] = pos

    def add_road(self, start, end, length=None, capacity=10, bidirectional=False):
        # Auto-calculate Euclidean distance if length not provided
        if length is None:
            p1 = self.positions[start]
            p2 = self.positions[end]
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        self.graph.add_edge(
            start,
            end,
            length=length,
            capacity=capacity,
            traffic=0
        )

        if bidirectional:
            self.graph.add_edge(
                end,
                start,
                length=length,
                capacity=capacity,
                traffic=0
            )

    def dynamic_shortest_path(self, start, end, globally_congested=None):
        if globally_congested is None:
            globally_congested = []

        def weight(u, v, d):
            # Local congestion (traffic / capacity)
            # We use a 2.0 multiplier to make congestion more impactful on routing
            congestion = d["traffic"] / d["capacity"]
            
            # Global congestion penalty (from server alerts)
            # If the target node 'v' is globally congested, apply a 100x penalty
            penalty = 100.0 if v in globally_congested else 1.0
            
            return d["length"] * (1 + congestion * 2.0) * penalty

        return nx.shortest_path(self.graph, start, end, weight=weight)

    def increase_traffic(self, start, end):
        self.graph[start][end]["traffic"] += 1

    def decay_traffic(self, rate=0.01):
        for u, v, data in self.graph.edges(data=True):
            if data["traffic"] > 0:
                data["traffic"] = max(0, data["traffic"] - rate)