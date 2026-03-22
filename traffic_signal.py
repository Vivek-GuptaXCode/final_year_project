import random


class TrafficSignal:
    def __init__(self, node, network):
        self.node = node
        self.network = network

        # self.timer = 0
        # self.phase = 0  # 0-5 phases
        self.phase = random.randint(0,3)
        self.timer = random.randint(0,200)
        # # timings (frames at 60 FPS)
        # self.green_time = 200
        # self.yellow_time = 60
        # self.all_red_time = 520

        FPS = 40

        # timings (seconds)
        green_seconds = 10
        yellow_seconds = 3
        all_red_seconds = 2

        # convert to frames
        self.green_time = green_seconds * FPS
        self.yellow_time = yellow_seconds * FPS
        self.all_red_time = all_red_seconds * FPS
        self.state = {}

        for u, v in network.graph.in_edges(node):
            self.state[(u, v)] = "RED"

    def update(self):
        self.timer += 1

        # ---- Phase switching ----
        if self.phase in [0, 2]:  # green phases
            if self.timer >= self.green_time:
                self.phase += 1
                self.timer = 0

        elif self.phase in [1, 3]:  # yellow phases
            if self.timer >= self.yellow_time:
                self.phase = (self.phase + 1) % 4
                self.timer = 0

        # ---- Reset all lanes to RED ----
        for key in self.state:
            self.state[key] = "RED"

        # ---- Assign lane states ----
        for u, v in self.network.graph.in_edges(self.node):

            start = self.network.positions[u]
            end = self.network.positions[v]

            dx = end[0] - start[0]
            dy = end[1] - start[1]

            is_vertical = abs(dy) > abs(dx)

            # Phase 0: Vertical GREEN
            if self.phase == 0 and is_vertical:
                self.state[(u, v)] = "GREEN"

            # Phase 1: Vertical YELLOW
            elif self.phase == 1 and is_vertical:
                self.state[(u, v)] = "YELLOW"

            # Phase 2: Horizontal GREEN
            elif self.phase == 2 and not is_vertical:
                self.state[(u, v)] = "GREEN"

            # Phase 3: Horizontal YELLOW
            elif self.phase == 3 and not is_vertical:
                self.state[(u, v)] = "YELLOW"