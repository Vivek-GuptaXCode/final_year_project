import os

import pygame
from network import RoadNetwork
from traffic_signal import TrafficSignal
from rsu import RSU
from vehicle import Vehicle
from rsu_client import rsu_network_client as rnc
import math
import random
import time
import metrics
import csv
from dotenv import load_dotenv
load_dotenv()
SERVER_URL=os.getenv("SERVER_URL")

WIDTH, HEIGHT = 1400, 700
NODE_RADIUS = 20
FPS = 40
#last_network_print = 0
vehicle_counter = 0
spawn_timer = 0
SPAWN_INTERVAL = 25

vehicles = []
SOURCES = ["A", "B"]
DESTINATIONS = ["K", "L", "I"]

# ---------------- CAMERA SETTINGS ----------------
zoom = 1.0
offset_x = 0
offset_y = 0
dragging = False
last_mouse_pos = (0, 0)
# -------------------------------------------------

pygame.init()
pygame.font.init()
font     = pygame.font.SysFont("Courier New", 14, bold=True)
hud_font = pygame.font.SysFont("Courier New", 13, bold=True)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("V2X Traffic Simulator")
clock = pygame.time.Clock()

# ---------------- THEME COLORS ----------------
BG_COLOR             = (8,  12,  20)
GRID_COLOR           = (15, 22,  35)
ROAD_GLOW_COLOR      = (25, 45,  75)
ROAD_COLOR           = (35, 50,  70)
ROAD_EDGE_COLOR      = (55, 75,  100)
NODE_FILL            = (18, 28,  45)
NODE_BORDER          = (60, 140, 200)
NODE_GLOW            = (25, 65,  115)
NODE_TEXT            = (120,190, 255)
RSU_COLOR            = (0,  180, 255)
SIGNAL_GREEN         = (0,  255, 140)
SIGNAL_YELLOW        = (255,210, 0)
SIGNAL_RED           = (255, 50, 80)
VEHICLE_CONNECTED    = (0,  230, 120)
VEHICLE_DISCONNECTED = (255, 60, 80)
HUD_COLOR            = (80, 160, 220)
HUD_BG               = (10, 18,  32)
CONGESTION_COLOR     = (255, 80,  0)
# ----------------------------------------------

# ---------------- NETWORK SETUP ----------------
network = RoadNetwork()

network.add_node("A",  (200,  2800))
network.add_node("B",  (1000, 2800))
network.add_node("C",  (200,  2400))
network.add_node("D",  (1000, 2400))
network.add_node("E",  (200,  2000))
network.add_node("F",  (1000, 2000))
network.add_node("M1", (600,  2000))
network.add_node("G",  (200,  1600))
network.add_node("H",  (1000, 1600))
network.add_node("M2", (600,  1600))
network.add_node("I",  (200,  1200))
network.add_node("J",  (1000, 800))
network.add_node("K",  (1000, 400))
network.add_node("L",  (1800, 800))

network.add_road("B",  "D",  bidirectional=True)
network.add_road("A",  "C",  bidirectional=True)
network.add_road("C",  "D",  bidirectional=True)
network.add_road("C",  "E",  bidirectional=True)
network.add_road("D",  "F",  bidirectional=True)
network.add_road("E",  "G",  bidirectional=True)
network.add_road("F",  "H",  bidirectional=True)
network.add_road("I",  "J",  bidirectional=True)
network.add_road("I",  "G",  bidirectional=True)
network.add_road("G",  "M2", bidirectional=True)
network.add_road("F",  "M1", bidirectional=True)
network.add_road("M1", "M2", bidirectional=True)
network.add_road("H",  "M2", bidirectional=True)
network.add_road("E",  "M1", bidirectional=True)
network.add_road("M2", "J",  bidirectional=True)
network.add_road("J",  "K",  bidirectional=True)
network.add_road("J",  "L",  bidirectional=True)
network.add_road("H",  "J",  bidirectional=True)

network.vehicles = vehicles

all_nodes = list(network.graph.nodes())

# --------- END NODES ----------
end_nodes = []
for node in network.graph.nodes():
    if network.graph.degree(node) == 2:
        end_nodes.append(node)
print("End Nodes:", end_nodes)

# Artificial congestion
for _ in range(5):
    network.increase_traffic("B", "D")

# --------- SIGNALS ----------
signals = []
for node in network.graph.nodes():
    neighbors = set()
    for u, v in network.graph.in_edges(node):
        neighbors.add(u)
    for u, v in network.graph.out_edges(node):
        neighbors.add(v)
    if len(neighbors) > 2:
        signals.append(TrafficSignal(node, network))
print("Signalized Junctions:", [s.node for s in signals])

# --------- RSUs ----------
# rsus = []
# for signal in signals:
#     rsu = RSU(signal.node, network)
#     rsu.signal = signal
#     rsus.append(rsu)
rsus = []
internal_nodes = [n for n in network.graph.nodes() if n not in SOURCES and n not in DESTINATIONS]
for node in internal_nodes:
    rsu = RSU(node, network)
    # Link signal if it exists
    rsu.signal = next((s for s in signals if s.node == node), None)
    rsus.append(rsu)
# --------- V2X SERVER CONNECTION ----------
#SERVER_URL = "https://192.168.137.6:5000"
rnc.connect(SERVER_URL)
time.sleep(1.0)   # give socket a moment to handshake

# Build RSU graph edges: two RSUs are connected if there is a road path
# between their junction nodes (direct neighbor check in the road graph).
rsu_nodes = [r.node for r in rsus]
rsu_edges = []
for i, u in enumerate(rsu_nodes):
    for v in rsu_nodes[i + 1:]:
        # Edge exists if both nodes are direct road-graph neighbors of each other
        if network.graph.has_edge(u, v) or network.graph.has_edge(v, u):
            rsu_edges.append([u, v])

rnc.register_rsus(rsu_nodes, rsu_edges)
print(f"[SIM] RSU nodes registered: {rsu_nodes}")
print(f"[SIM] RSU graph edges:      {rsu_edges}")
# ---- CENTER CAMERA ON NETWORK ----
all_x = [pos[0] for pos in network.positions.values()]
all_y = [pos[1] for pos in network.positions.values()]

net_cx = (min(all_x) + max(all_x)) / 2
net_cy = (min(all_y) + max(all_y)) / 2

zoom = 0.25  # fit large network into window

offset_x = WIDTH  / 2 - net_cx * zoom
offset_y = HEIGHT / 2 - net_cy * zoom

# --------- CONGESTION STATE (for glow) ----------
congested_nodes = set()
# Linger timer: how many frames a node stays visually congested after last
# real detection.  Prevents rapid blinking when score hovers near threshold.
CONGESTION_LINGER_FRAMES = 50   # ~6 s at 15 FPS, ~1.5 s at 60 FPS
congestion_linger = {}           # node -> frames_remaining


# ---------------- HELPER: draw glow circle ----------------
def draw_glow_circle(surface, color, center, radius, alpha=60, layers=3):
    for i in range(layers, 0, -1):
        r = radius + i * 4
        glow_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        glow_alpha = alpha // i
        pygame.draw.circle(glow_surf, (*color, glow_alpha), (r, r), r)
        surface.blit(glow_surf, (center[0] - r, center[1] - r))

# ---------------- NETWORK TIMER ----------------
start_time = time.time()

# store networking metrics
network_log = []

# optional: control printing
last_network_print = 0
# ---------------- MAIN LOOP ----------------

running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEWHEEL:
            zoom += event.y * 0.1
            zoom = max(0.3, min(3.0, zoom))

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
                last_mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False

        if event.type == pygame.MOUSEMOTION:
            if dragging:
                mx, my = pygame.mouse.get_pos()
                dx = mx - last_mouse_pos[0]
                dy = my - last_mouse_pos[1]
                offset_x += dx
                offset_y += dy
                last_mouse_pos = (mx, my)

    # ---- UPDATE SIGNALS ----
    for signal in signals:
        signal.update()

    # ---- SPAWN VEHICLES ----
    spawn_timer += 1
    if spawn_timer > SPAWN_INTERVAL and len(vehicles) < 100:
        spawn_timer = 0
        vehicle_counter += 1
        start       = random.choice(SOURCES)
        destination = random.choice(DESTINATIONS)
        while destination == start:
            destination = random.choice(DESTINATIONS)
        try:
            vehicles.append(Vehicle(vehicle_counter, network, start, destination))
        except:
            pass

    # ---- RSU CONGESTION CHECK ----
    for rsu in rsus:
        # Only the new wait-based aggregation logic is needed
        if rsu.node == "M1":
            rnc.send_junction_congestion_alert("M1", 10, 999)
            # rnc.send_junction_congestion_alert("E", 10, 999)
            # rnc.send_junction_congestion_alert("F", 10, 999)
        rsu.check_long_wait_batch(client=rnc)

    # Fetch globally congested nodes from the server for routing and display
    globally_congested = rnc.get_active_congested_nodes()

    # ---- SYNC VISUAL STATE ----
    congested_nodes.clear()
    
    # Prune explicitly cleared nodes from linger immediately
    for node in rnc.consume_cleared_nodes():
        congestion_linger.pop(node, None)

    for node in globally_congested:
        congested_nodes.add(node)
        # Reset (or start) the linger countdown
        congestion_linger[node] = CONGESTION_LINGER_FRAMES

    # ---- LINGER: keep visually congested for N frames after last detection ----
    for node in list(congestion_linger):
        if congestion_linger[node] > 0:
            congested_nodes.add(node)      # stay orange
            congestion_linger[node] -= 1
        else:
            del congestion_linger[node]    # linger expired → back to normal

    # Decay traffic over time so congestion can clear
    network.decay_traffic(rate=0.005)

    # ================================================================
    #  DRAW
    # ================================================================

    # --- Background ---
    screen.fill(BG_COLOR)

    # --- Grid ---
    grid_spacing = int(80 * zoom)
    if grid_spacing > 8:
        start_x = int(offset_x % grid_spacing)
        for x in range(start_x, WIDTH, grid_spacing):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT), 1)
        start_y = int(offset_y % grid_spacing)
        for y in range(start_y, HEIGHT, grid_spacing):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y), 1)

    # --- RSU ranges (draw first, behind roads) ---
    for rsu in rsus:
        pos = rsu.get_position()
        sx  = int(pos[0] * zoom + offset_x)
        sy  = int(pos[1] * zoom + offset_y)
        sr  = int(rsu.radius * zoom)

        if sr > 0:
            # Transparent fill
            rsu_surf = pygame.Surface((sr * 2, sr * 2), pygame.SRCALPHA)
            fill_color = (255, 80, 0, 15) if rsu.node in congested_nodes else (0, 180, 255, 12)
            pygame.draw.circle(rsu_surf, fill_color, (sr, sr), sr)
            screen.blit(rsu_surf, (sx - sr, sy - sr))

            # Ring
            ring_color = CONGESTION_COLOR if rsu.node in congested_nodes else RSU_COLOR
            pygame.draw.circle(screen, ring_color, (sx, sy), sr, 1)

    # --- Roads ---
    for u, v in network.graph.edges():
        sp = network.positions[u]
        ep = network.positions[v]

        sx1 = sp[0] * zoom + offset_x
        sy1 = sp[1] * zoom + offset_y
        sx2 = ep[0] * zoom + offset_x
        sy2 = ep[1] * zoom + offset_y

        dx = sx2 - sx1
        dy = sy2 - sy1
        length = math.hypot(dx, dy)
        if length == 0:
            continue

        ox = -dy / length * 5
        oy =  dx / length * 5

        lane_s = (sx1 + ox, sy1 + oy)
        lane_e = (sx2 + ox, sy2 + oy)

        # Glow layer
        pygame.draw.line(screen, ROAD_GLOW_COLOR, lane_s, lane_e, max(2, int(9 * zoom)))
        # Road body
        pygame.draw.line(screen, ROAD_COLOR,      lane_s, lane_e, max(1, int(5 * zoom)))
        # Edge highlight
        pygame.draw.line(screen, ROAD_EDGE_COLOR, lane_s, lane_e, 1)

    # --- Nodes ---
    for node, pos in network.positions.items():
        sx = int(pos[0] * zoom + offset_x)
        sy = int(pos[1] * zoom + offset_y)
        r  = int(NODE_RADIUS * zoom)

        is_local_congested = node in congested_nodes
        is_global_congested = node in globally_congested
        
        # Decide border and glow colors
        if is_local_congested:
            border_color = CONGESTION_COLOR 
            glow_color   = (180, 50, 0)
        elif is_global_congested:
            border_color = (255, 180, 0) # Lighter orange for "Informed" congestion
            glow_color   = (120, 90, 0)
        else:
            border_color = NODE_BORDER
            glow_color   = (30, 70, 120)

        # Glow ring
        pygame.draw.circle(screen, glow_color,   (sx, sy), r + max(1, int(4 * zoom)))
        # Fill
        pygame.draw.circle(screen, NODE_FILL,    (sx, sy), r)
        # Border
        pygame.draw.circle(screen, border_color, (sx, sy), r, max(1, int(2 * zoom)))

        # Label
        text_surf = font.render(str(node), True, NODE_TEXT)
        text_rect = text_surf.get_rect(center=(sx, sy))
        screen.blit(text_surf, text_rect)

    # --- Traffic Signals ---
    for signal in signals:
        for (u, v), state in signal.state.items():
            sp = network.positions[u]
            ep = network.positions[v]

            dx = ep[0] - sp[0]
            dy = ep[1] - sp[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            dir_x = dx / length
            dir_y = dy / length
            px = ep[0] - dir_x * NODE_RADIUS
            py = ep[1] - dir_y * NODE_RADIUS

            sx = int(px * zoom + offset_x)
            sy = int(py * zoom + offset_y)
            sr = max(3, int(5 * zoom))

            color = (SIGNAL_GREEN  if state == "GREEN"  else
                     SIGNAL_YELLOW if state == "YELLOW" else
                     SIGNAL_RED)

            # Glow halo
            glow_surf = pygame.Surface((sr * 8, sr * 8), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (sr * 4, sr * 4), sr * 4)
            screen.blit(glow_surf, (sx - sr * 4, sy - sr * 4))

            # Signal dot
            pygame.draw.circle(screen, color, (sx, sy), sr)
            # Inner bright core
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), max(1, sr // 2))

    # --- Vehicles ---
    for v in vehicles:
        v.update(signals, vehicles, rsus, globally_congested=globally_congested)
        v.draw(screen, zoom, offset_x, offset_y)

    # ---------------- NETWORK METRICS ----------------

    total_sent = metrics.global_packets_sent
    total_received = metrics.global_packets_received

    pdr = total_received / total_sent if total_sent else 0

    sim_time = time.time() - start_time

    packet_rate = total_sent / sim_time if sim_time else 0

    connected_count = sum(
        1 for v in vehicles if v.obu.connected_rsu is not None
    )

    # Throughput approximation (bytes/sec)
    avg_packet_size = 200
    throughput = (total_received * avg_packet_size) / sim_time if sim_time else 0


    # store log for graph generation
    network_log.append({
        "time": sim_time,
        "sent": total_sent,
        "received": total_received,
        "pdr": pdr,
        "throughput": throughput,
        "packet_rate": packet_rate,
        "connected": connected_count
    })


    # print every 5 seconds
    now = time.time()
    if now - last_network_print > 5:

        print("------ NETWORK METRICS ------")
        print("Packets Sent:", total_sent)
        print("Packets Received:", total_received)
        print("Packet Delivery Ratio:", round(pdr,3))
        print("Throughput:", round(throughput,2), "bytes/sec")
        print("Packet Rate:", round(packet_rate,2), "pkts/sec")
        print("Connected Vehicles:", connected_count)

        last_network_print = now
    # Remove finished vehicles
    vehicles[:] = [v for v in vehicles if v.target_node is not None]

    # --- HUD ---
    connected_count = sum(1 for v in vehicles if v.obu.connected_rsu is not None)
    congestion_count = len(congested_nodes)

    hud_lines = [
        f"VEHICLES   {len(vehicles):>4}",
        f"CONNECTED  {connected_count:>4}",
        f"RSUs       {len(rsus):>4}",
        f"CONGESTED  {congestion_count:>4}",
        f"ZOOM       {zoom:>5.2f}x",
        f"FPS        {int(clock.get_fps()):>4}",
    ]

    # HUD background panel
    panel_w = 160
    panel_h = len(hud_lines) * 20 + 16
    panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel_surf.fill((8, 14, 28, 200))
    pygame.draw.rect(panel_surf, (40, 100, 160, 180), (0, 0, panel_w, panel_h), 1)
    screen.blit(panel_surf, (8, 8))

    # HUD title
    title_surf = hud_font.render("[ V2X MONITOR ]", True, (60, 140, 220))
    screen.blit(title_surf, (14, 12))

    for i, line in enumerate(hud_lines):
        surf = hud_font.render(line, True, HUD_COLOR)
        screen.blit(surf, (14, 30 + i * 18))

    # --- Legend ---
    legend_items = [
        (VEHICLE_CONNECTED,    "RSU Connected"),
        (VEHICLE_DISCONNECTED, "No Coverage"),
        (CONGESTION_COLOR,     "Congested"),
        (RSU_COLOR,            "RSU Range"),
    ]
    leg_x = WIDTH - 155
    leg_y = 12
    leg_w, leg_h = 148, len(legend_items) * 20 + 16

    leg_surf = pygame.Surface((leg_w, leg_h), pygame.SRCALPHA)
    leg_surf.fill((8, 14, 28, 200))
    pygame.draw.rect(leg_surf, (40, 100, 160, 180), (0, 0, leg_w, leg_h), 1)
    screen.blit(leg_surf, (leg_x - 4, leg_y))

    for i, (color, label) in enumerate(legend_items):
        cy = leg_y + 10 + i * 20
        pygame.draw.circle(screen, color, (leg_x + 8, cy), 5)
        lbl = hud_font.render(label, True, HUD_COLOR)
        screen.blit(lbl, (leg_x + 20, cy - 7))

    pygame.display.flip()
if len(network_log) > 0:
    with open("network_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=network_log[0].keys())
        writer.writeheader()
        writer.writerows(network_log)

    print("Saved network results to network_results.csv")
pygame.quit()