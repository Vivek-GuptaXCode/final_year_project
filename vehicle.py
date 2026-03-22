# import pygame
# import math
# import random

# VEHICLE_RADIUS = 6
# NODE_RADIUS = 20


# class OBU:
#     """On-Board Unit — transmits telemetry to RSU."""
#     def __init__(self, vehicle):
#         self.vehicle = vehicle
#         self.connected_rsu = None

#     def get_telemetry(self):
#         """Compute and return current telemetry packet."""
#         v = self.vehicle

#         if v.target_node is None:
#             return None

#         start_pos = v.network.positions[v.current_node]
#         end_pos = v.network.positions[v.target_node]

#         dx = end_pos[0] - start_pos[0]
#         dy = end_pos[1] - start_pos[1]
#         length = math.hypot(dx, dy)

#         if length == 0:
#             return None

#         # GPS position
#         gps_x = start_pos[0] + dx * v.progress
#         gps_y = start_pos[1] + dy * v.progress

#         # Heading (angle in degrees)
#         heading = math.degrees(math.atan2(dy, dx))

#         # Acceleration (speed change since last frame)
#         acceleration = v.current_speed - v.prev_speed

#         # Local vehicle density (vehicles within 150 units)
#         nearby = 0
#         for other in v.network.vehicles:
#             if other is v or other.target_node is None:
#                 continue
#             o_start = other.network.positions[other.current_node]
#             o_end = other.network.positions[other.target_node]
#             odx = o_end[0] - o_start[0]
#             ody = o_end[1] - o_start[1]
#             ox = o_start[0] + odx * other.progress
#             oy = o_start[1] + ody * other.progress
#             if math.hypot(ox - gps_x, oy - gps_y) <= 150:
#                 nearby += 1

#         return {
#             "vehicle_id": v.id,
#             "gps": (gps_x, gps_y),
#             "speed": v.current_speed,
#             "heading": heading,
#             "acceleration": acceleration,
#             "density_estimate": nearby
#         }

#     def get_signal_strength(self, rsu):
#         """Signal strength = inverse of distance to RSU (0 if out of range)."""
#         telemetry = self.get_telemetry()
#         if telemetry is None:
#             return 0

#         gps_x, gps_y = telemetry["gps"]
#         rsu_x, rsu_y = rsu.get_position()
#         dist = math.hypot(gps_x - rsu_x, gps_y - rsu_y)

#         if dist > rsu.radius:
#             return 0

#         return 1 / (dist + 1e-6)  # stronger when closer

#     def handover(self, rsus):
#         """Switch to RSU with strongest signal if better than current."""
#         best_rsu = None
#         best_strength = 0

#         for rsu in rsus:
#             strength = self.get_signal_strength(rsu)
#             if strength > best_strength:
#                 best_strength = strength
#                 best_rsu = rsu

#         # Handover if new RSU is better
#         if best_rsu is not self.connected_rsu:
#             if self.connected_rsu is not None:
#                 self.connected_rsu.remove_vehicle(self.vehicle)
#             self.connected_rsu = best_rsu
#             if best_rsu is not None:
#                 best_rsu.register_vehicle(self.vehicle)

#     def transmit(self):
#         """Send telemetry to connected RSU."""
#         if self.connected_rsu is None:
#             return
#         telemetry = self.get_telemetry()
#         if telemetry:
#             self.connected_rsu.receive_telemetry(self.vehicle, telemetry)

#     def send_long_wait_notification(self, wait_duration):
#         """Notify the RSU that the vehicle has been stuck."""
#         if self.connected_rsu is None:
#             return
        
#         # We don't send the full telemetry again, just the wait info
#         self.connected_rsu.receive_long_wait_notification(self.vehicle, wait_duration)


# class Vehicle:
#     def __init__(self, vid, network, start, destination):
#         self.id = vid
#         self.network = network
#         self.current_node = start
#         self.destination = destination
        
#         # In the "RSU Instruction" model, the vehicle doesn't have a route.
#         # It just knows which node it is driving towards.
#         self.target_node = None 
#         self.progress = 0

#         # Standardized speed for more predictable flow
#         self.base_speed = random.uniform(0.002, 0.004)
#         self.current_speed = self.base_speed
#         self.prev_speed = self.base_speed

#         self.color = (0, 0, 255)
#         self.obu = OBU(self)

#         # ---- Enhanced Wait Tracking ----
#         self.wait_time = 0
#         self.move_time = 0                 # frames spent moving "fast"
#         self.next_notify_threshold = 200   # start with 5s at 40FPS
#         self.max_notify_threshold = 2400   # cap at 1 minute

#     def update(self, signals, vehicles, rsus=None, globally_congested=None):
#         # Initial instruction if we don't have a target yet
#         rsus = rsus or []
#         if self.target_node is None:
#             if self.current_node == self.destination:
#                 return # Arrived
                
#             # ── RSU NAVIGATION HELP ──
#             # rsu = next((r for r in rsus if r.node == self.current_node), None)
#             rsu = None
#             if rsus:
#                 rsu = next((r for r in rsus if r.node == self.current_node), None)
#             if rsu:
#                 self.target_node = rsu.get_next_hop(self.destination, globally_congested)
#             else:
#                 # Fallback to direct shortest path if no RSU is present
#                 try:
#                     path = self.network.dynamic_shortest_path(self.current_node, self.destination, globally_congested)
#                     if len(path) > 1:
#                         self.target_node = path[1]
#                 except:
#                     pass
        
#         if self.target_node is None:
#             return

#         speed = self.base_speed
#         self.prev_speed = self.current_speed

#         # ---- TRAFFIC SIGNAL CHECK ----
#         for signal in signals:
#             if signal.node == self.target_node:
#                 lane_state = signal.state.get((self.current_node, self.target_node), "GREEN")
#                 if lane_state in ["RED", "YELLOW"]:
#                     start_pos = self.network.positions[self.current_node]
#                     end_pos = self.network.positions[self.target_node]
#                     dx = end_pos[0] - start_pos[0]
#                     dy = end_pos[1] - start_pos[1]
#                     length = math.hypot(dx, dy)
#                     if length == 0:
#                         continue
#                     stop_distance = NODE_RADIUS + 10
#                     stop_progress = 1 - (stop_distance / length)
#                     if self.progress >= stop_progress:
#                         speed = 0

#         # ---- PREVENT OVERLAPPING (QUEUE MODEL) ----
#         for other in vehicles:
#             if other is self:
#                 continue
#             if (other.current_node == self.current_node and
#                     other.target_node == self.target_node):
#                 if other.progress > self.progress:
#                     gap = other.progress - self.progress
#                     # Scaled gap check for realistic spacing
#                     if gap < 0.03: 
#                         speed = 0

#         # ---- WAIT TRACKING LOGIC ----
#         if speed < 0.001:  # "Slow-moving" threshold
#             self.wait_time += 1
#             self.move_time = 0
#         else:
#             self.move_time += 1
#             # Persistence: Only reset wait_time if moving for > 2 seconds
#             if self.move_time > 80:
#                 self.wait_time = 0
#                 self.next_notify_threshold = 200 # reset backoff

#         # ---- LONG WAIT NOTIFICATION (Exponential Backoff) ----
#         if self.wait_time >= self.next_notify_threshold:
#             self.obu.send_long_wait_notification(self.wait_time)
#             # Increase next threshold (exponentially)
#             self.next_notify_threshold = min(
#                 int(self.next_notify_threshold * 1.5), 
#                 self.max_notify_threshold
#             )

#         self.current_speed = speed

#         # ---- OBU: HANDOVER + TRANSMIT ----
#         if rsus:
#             self.obu.handover(rsus)
#             self.obu.transmit()

#         self.progress += speed

#         if self.progress >= 1:
#             # We just reached 'target_node'
#             self.current_node = self.target_node
#             self.progress = 0
            
#             if self.current_node == self.destination:
#                 self.target_node = None
#                 return

#             # Request next instruction from the RSU at the new node
#             rsu = next((r for r in rsus if r.node == self.current_node), None)
#             if rsu:
#                 self.target_node = rsu.get_next_hop(self.destination, globally_congested)
#             else:
#                 # Fallback to direct shortest path
#                 try:
#                     path = self.network.dynamic_shortest_path(self.current_node, self.destination, globally_congested)
#                     if len(path) > 1:
#                         self.target_node = path[1]
#                     else:
#                         self.target_node = None
#                 except:
#                     self.target_node = None

#             if self.target_node:
#                 self.network.increase_traffic(self.current_node, self.target_node)

#     def draw(self, screen, zoom, offset_x, offset_y):

#         if self.target_node is None:
#             return

#         start_pos = self.network.positions[self.current_node]
#         end_pos = self.network.positions[self.target_node]

#         dx = end_pos[0] - start_pos[0]
#         dy = end_pos[1] - start_pos[1]

#         length = math.hypot(dx, dy)

#         if length == 0:
#             return

#         lane_offset_x = -dy / length * 5
#         lane_offset_y = dx / length * 5

#         x = start_pos[0] + dx * self.progress + lane_offset_x
#         y = start_pos[1] + dy * self.progress + lane_offset_y

#         screen_x = x * zoom + offset_x
#         screen_y = y * zoom + offset_y

#         angle = math.atan2(dy, dx)

#         size = 8 * zoom

#         front = (
#             screen_x + math.cos(angle) * size,
#             screen_y + math.sin(angle) * size
#         )

#         left = (
#             screen_x + math.cos(angle + 2.5) * size,
#             screen_y + math.sin(angle + 2.5) * size
#         )

#         right = (
#             screen_x + math.cos(angle - 2.5) * size,
#             screen_y + math.sin(angle - 2.5) * size
#         )

#         if self.obu.connected_rsu:
#             color = (0, 200, 0)
#         else:
#             color = (200, 0, 0)

#         pygame.draw.polygon(screen, color, [front, left, right])


# import pygame
# import math
# import random
# import time
# import json
# import struct
# import zlib




# VEHICLE_RADIUS = 6
# NODE_RADIUS = 20


# # ---------------------------------------------------------
# # MAC Address Generator
# # ---------------------------------------------------------
# def generate_mac():
#     return "02:%02x:%02x:%02x:%02x:%02x" % tuple(
#         random.randint(0, 255) for _ in range(5)
#     )


# # ---------------------------------------------------------
# # IEEE 802.11p Frame
# # ---------------------------------------------------------
# class IEEE80211pFrame:

#     def __init__(self, receiver_mac, sender_mac, payload, sequence=0):

#         self.frame_control = 0x0801
#         self.duration = 0
#         self.receiver_mac = receiver_mac
#         self.sender_mac = sender_mac
#         self.bssid = "00:00:00:00:00:00"
#         self.sequence = sequence
#         self.payload = payload

#     def mac_to_bytes(self, mac):
#         return bytes(int(x, 16) for x in mac.split(":"))

#     def build(self):

#         header = struct.pack("!HH", self.frame_control, self.duration)

#         header += self.mac_to_bytes(self.receiver_mac)
#         header += self.mac_to_bytes(self.sender_mac)
#         header += self.mac_to_bytes(self.bssid)

#         header += struct.pack("!H", self.sequence)

#         frame = header + self.payload

#         crc = zlib.crc32(frame) & 0xffffffff

#         frame += struct.pack("!I", crc)

#         return frame


# # ---------------------------------------------------------
# # OBU
# # ---------------------------------------------------------
# class OBU:
#     """On-Board Unit — transmits telemetry to RSU."""

#     def __init__(self, vehicle):
#         self.packets_sent = 0
#         self.vehicle = vehicle
#         self.connected_rsu = None

#         # IEEE 802.11p transmission parameters
#         self.last_tx_time = 0
#         self.tx_interval = random.uniform(0.1, 1.0)  # 1–10 messages/sec

#     # -----------------------------------------------------
#     # TELEMETRY
#     # -----------------------------------------------------
#     def get_telemetry(self):

#         v = self.vehicle

#         if v.target_node is None:
#             return None

#         start_pos = v.network.positions[v.current_node]
#         end_pos = v.network.positions[v.target_node]

#         dx = end_pos[0] - start_pos[0]
#         dy = end_pos[1] - start_pos[1]
#         length = math.hypot(dx, dy)

#         if length == 0:
#             return None

#         gps_x = start_pos[0] + dx * v.progress
#         gps_y = start_pos[1] + dy * v.progress

#         heading = math.degrees(math.atan2(dy, dx))

#         acceleration = v.current_speed - v.prev_speed

#         nearby = 0

#         for other in v.network.vehicles:

#             if other is v or other.target_node is None:
#                 continue

#             o_start = other.network.positions[other.current_node]
#             o_end = other.network.positions[other.target_node]

#             odx = o_end[0] - o_start[0]
#             ody = o_end[1] - o_start[1]

#             ox = o_start[0] + odx * other.progress
#             oy = o_start[1] + ody * other.progress

#             if math.hypot(ox - gps_x, oy - gps_y) <= 150:
#                 nearby += 1

#         return {
#             "vehicle_id": v.id,
#             "gps": (gps_x, gps_y),
#             "speed": v.current_speed,
#             "heading": heading,
#             "acceleration": acceleration,
#             "density_estimate": nearby,
#             "wait_time": v.wait_time,
#             "timestamp": time.time()
#         }

#     # -----------------------------------------------------
#     # SIGNAL STRENGTH
#     # -----------------------------------------------------
#     def get_signal_strength(self, rsu):

#         telemetry = self.get_telemetry()

#         if telemetry is None:
#             return 0

#         gps_x, gps_y = telemetry["gps"]

#         rsu_x, rsu_y = rsu.get_position()

#         dist = math.hypot(gps_x - rsu_x, gps_y - rsu_y)

#         if dist > rsu.radius:
#             return 0

#         return 1 / (dist + 1e-6)

#     # -----------------------------------------------------
#     # RSU HANDOVER
#     # -----------------------------------------------------
#     def handover(self, rsus):

#         best_rsu = None
#         best_strength = 0

#         for rsu in rsus:

#             strength = self.get_signal_strength(rsu)

#             if strength > best_strength:
#                 best_strength = strength
#                 best_rsu = rsu

#         if best_rsu is not self.connected_rsu:

#             if self.connected_rsu is not None:
#                 self.connected_rsu.remove_vehicle(self.vehicle)

#             self.connected_rsu = best_rsu

#             if best_rsu is not None:
#                 best_rsu.register_vehicle(self.vehicle)

#     # -----------------------------------------------------
#     # IEEE 802.11p TRANSMISSION
#     # -----------------------------------------------------
#     def transmit(self):
        
#         if self.connected_rsu is None:
#             return

#         now = time.time()

#         # enforce 1–10 messages/sec
#         if now - self.last_tx_time < self.tx_interval:
#             return

#         telemetry = self.get_telemetry()

#         if telemetry is None:
#             return

#         payload = json.dumps(telemetry).encode()

#         frame = IEEE80211pFrame(
#             receiver_mac=self.connected_rsu.mac,
#             sender_mac=self.vehicle.mac,
#             payload=payload,
#             sequence=int(now * 1000) % 4096
#         )

#         raw_frame = frame.build()

#         # Simulated DSRC latency (10–100 ms)
#         # latency = random.uniform(0.01, 0.1)
#         # time.sleep(latency)
#         # Simulated DSRC latency (0–5 ms)
#         latency = random.uniform(0.0, 0.005)
#         time.sleep(latency)
#         self.packets_sent += 1
#         # Deliver frame to RSU

      
#         import metrics
#         metrics.global_packets_sent += 1

#         self.connected_rsu.receive_frame(self.vehicle, raw_frame)

#         self.last_tx_time = now

#     # -----------------------------------------------------
#     # LONG WAIT ALERT
#     # -----------------------------------------------------
#     def send_long_wait_notification(self, wait_duration):

#         if self.connected_rsu is None:
#             return

#         self.connected_rsu.receive_long_wait_notification(
#             self.vehicle,
#             wait_duration
#         )


# # ---------------------------------------------------------
# # VEHICLE
# # ---------------------------------------------------------
# class Vehicle:

#     def __init__(self, vid, network, start, destination):

#         self.id = vid
#         self.network = network
#         self.current_node = start
#         self.destination = destination

#         self.target_node = None
#         self.progress = 0

#         self.mac = generate_mac()

#         self.base_speed = random.uniform(0.002, 0.004)

#         self.current_speed = self.base_speed
#         self.prev_speed = self.base_speed

#         self.color = (0, 0, 255)

#         self.obu = OBU(self)

#         self.wait_time = 0
#         self.move_time = 0

#         self.next_notify_threshold = 200
#         self.max_notify_threshold = 2400
#     # -----------------------------------------------------
#     # UPDATE
#     # -----------------------------------------------------
#     def update(self, signals, vehicles, rsus=None, globally_congested=None):

#         rsus = rsus or []

#         if self.target_node is None:

#             if self.current_node == self.destination:
#                 return

#             rsu = None

#             if rsus:
#                 rsu = next((r for r in rsus if r.node == self.current_node), None)

#             if rsu:

#                 self.target_node = rsu.get_next_hop(
#                     self.destination,
#                     globally_congested
#                 )

#             else:

#                 try:
#                     path = self.network.dynamic_shortest_path(
#                         self.current_node,
#                         self.destination,
#                         globally_congested
#                     )

#                     if len(path) > 1:
#                         self.target_node = path[1]

#                 except:
#                     pass

#         if self.target_node is None:
#             return

#         speed = self.base_speed
#         self.prev_speed = self.current_speed

#         for signal in signals:

#             if signal.node == self.target_node:

#                 lane_state = signal.state.get(
#                     (self.current_node, self.target_node),
#                     "GREEN"
#                 )

#                 if lane_state in ["RED", "YELLOW"]:

#                     start_pos = self.network.positions[self.current_node]
#                     end_pos = self.network.positions[self.target_node]

#                     dx = end_pos[0] - start_pos[0]
#                     dy = end_pos[1] - start_pos[1]

#                     length = math.hypot(dx, dy)

#                     if length == 0:
#                         continue

#                     stop_distance = NODE_RADIUS + 10
#                     stop_progress = 1 - (stop_distance / length)

#                     if self.progress >= stop_progress:
#                         speed = 0

#         for other in vehicles:

#             if other is self:
#                 continue

#             if (
#                 other.current_node == self.current_node
#                 and other.target_node == self.target_node
#             ):

#                 if other.progress > self.progress:

#                     gap = other.progress - self.progress

#                     if gap < 0.03:
#                         speed = 0

#         if speed < 0.001:

#             self.wait_time += 1
#             self.move_time = 0

#         else:

#             self.move_time += 1

#             if self.move_time > 100:

#                 self.wait_time = 0
#                 self.next_notify_threshold = 200

#         if self.wait_time >= self.next_notify_threshold:

#             self.obu.send_long_wait_notification(self.wait_time)

#             self.next_notify_threshold = min(
#                 int(self.next_notify_threshold * 1.5),
#                 self.max_notify_threshold
#             )

#         self.current_speed = speed

#         if rsus:

#             self.obu.handover(rsus)
#             self.obu.transmit()

#         self.progress += speed

#         if self.progress >= 1:

#             self.current_node = self.target_node
#             self.progress = 0

#             if self.current_node == self.destination:

#                 self.target_node = None
#                 return

#             rsu = next((r for r in rsus if r.node == self.current_node), None)

#             if rsu:

#                 self.target_node = rsu.get_next_hop(
#                     self.destination,
#                     globally_congested
#                 )

#             else:

#                 try:

#                     path = self.network.dynamic_shortest_path(
#                         self.current_node,
#                         self.destination,
#                         globally_congested
#                     )

#                     if len(path) > 1:
#                         self.target_node = path[1]

#                 except:

#                     self.target_node = None

#             if self.target_node:

#                 self.network.increase_traffic(
#                     self.current_node,
#                     self.target_node
#                 )

#     # -----------------------------------------------------
#     # DRAW
#     # -----------------------------------------------------
#     def draw(self, screen, zoom, offset_x, offset_y):

#         if self.target_node is None:
#             return

#         start_pos = self.network.positions[self.current_node]
#         end_pos = self.network.positions[self.target_node]

#         dx = end_pos[0] - start_pos[0]
#         dy = end_pos[1] - start_pos[1]

#         length = math.hypot(dx, dy)

#         if length == 0:
#             return

#         # lane_offset_x = -dy / length * 5
#         # lane_offset_y = dx / length * 5

#         lane_width = 8
#         offset_multiplier = -2.5 if self.lane_id == 0 else 2.5
        
#         lane_offset_x = -dy / length * offset_multiplier
#         lane_offset_y = dx / length * offset_multiplier

#         x = start_pos[0] + dx * self.progress + lane_offset_x
#         y = start_pos[1] + dy * self.progress + lane_offset_y

#         screen_x = x * zoom + offset_x
#         screen_y = y * zoom + offset_y

#         angle = math.atan2(dy, dx)

#         size = 8 * zoom

#         front = (
#             screen_x + math.cos(angle) * size,
#             screen_y + math.sin(angle) * size
#         )

#         left = (
#             screen_x + math.cos(angle + 2.5) * size,
#             screen_y + math.sin(angle + 2.5) * size
#         )

#         right = (
#             screen_x + math.cos(angle - 2.5) * size,
#             screen_y + math.sin(angle - 2.5) * size
#         )

#         if self.obu.connected_rsu:
#             color = (0, 200, 0)
#         else:
#             color = (200, 0, 0)

#         pygame.draw.polygon(screen, color, [front, left, right])




import pygame
import math
import random
import time
import json
import struct
import zlib




VEHICLE_RADIUS = 3
NODE_RADIUS = 20


# ---------------------------------------------------------
# MAC Address Generator
# ---------------------------------------------------------
def generate_mac():
    return "02:%02x:%02x:%02x:%02x:%02x" % tuple(
        random.randint(0, 255) for _ in range(5)
    )


# ---------------------------------------------------------
# IEEE 802.11p Frame
# ---------------------------------------------------------
class IEEE80211pFrame:

    def __init__(self, receiver_mac, sender_mac, payload, sequence=0):

        self.frame_control = 0x0801
        self.duration = 0
        self.receiver_mac = receiver_mac
        self.sender_mac = sender_mac
        self.bssid = "00:00:00:00:00:00"
        self.sequence = sequence
        self.payload = payload

    def mac_to_bytes(self, mac):
        return bytes(int(x, 16) for x in mac.split(":"))

    def build(self):

        header = struct.pack("!HH", self.frame_control, self.duration)

        header += self.mac_to_bytes(self.receiver_mac)
        header += self.mac_to_bytes(self.sender_mac)
        header += self.mac_to_bytes(self.bssid)

        header += struct.pack("!H", self.sequence)

        frame = header + self.payload

        crc = zlib.crc32(frame) & 0xffffffff

        frame += struct.pack("!I", crc)

        return frame


# ---------------------------------------------------------
# OBU
# ---------------------------------------------------------
class OBU:
    """On-Board Unit — transmits telemetry to RSU."""

    def __init__(self, vehicle):
        self.packets_sent = 0
        self.vehicle = vehicle
        self.connected_rsu = None

        # IEEE 802.11p transmission parameters
        self.last_tx_time = 0
        self.tx_interval = random.uniform(0.1, 1.0)  # 1–10 messages/sec

    # -----------------------------------------------------
    # TELEMETRY
    # -----------------------------------------------------
    def get_telemetry(self):

        v = self.vehicle

        if v.target_node is None:
            return None

        start_pos = v.network.positions[v.current_node]
        end_pos = v.network.positions[v.target_node]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = math.hypot(dx, dy)

        if length == 0:
            return None

        gps_x = start_pos[0] + dx * v.progress
        gps_y = start_pos[1] + dy * v.progress

        heading = math.degrees(math.atan2(dy, dx))

        acceleration = v.current_speed - v.prev_speed

        nearby = 0

        for other in v.network.vehicles:

            if other is v or other.target_node is None:
                continue

            o_start = other.network.positions[other.current_node]
            o_end = other.network.positions[other.target_node]

            odx = o_end[0] - o_start[0]
            ody = o_end[1] - o_start[1]

            ox = o_start[0] + odx * other.progress
            oy = o_start[1] + ody * other.progress

            if math.hypot(ox - gps_x, oy - gps_y) <= 150:
                nearby += 1

        return {
            "vehicle_id": v.id,
            "gps": (gps_x, gps_y),
            "speed": v.current_speed,
            "heading": heading,
            "acceleration": acceleration,
            "density_estimate": nearby,
            "timestamp": time.time()
        }

    # -----------------------------------------------------
    # SIGNAL STRENGTH
    # -----------------------------------------------------
    def get_signal_strength(self, rsu):

        telemetry = self.get_telemetry()

        if telemetry is None:
            return 0

        gps_x, gps_y = telemetry["gps"]

        rsu_x, rsu_y = rsu.get_position()

        dist = math.hypot(gps_x - rsu_x, gps_y - rsu_y)

        if dist > rsu.radius:
            return 0

        return 1 / (dist + 1e-6)

    # -----------------------------------------------------
    # RSU HANDOVER
    # -----------------------------------------------------
    def handover(self, rsus):

        best_rsu = None
        best_strength = 0

        for rsu in rsus:

            strength = self.get_signal_strength(rsu)

            if strength > best_strength:
                best_strength = strength
                best_rsu = rsu

        if best_rsu is not self.connected_rsu:

            if self.connected_rsu is not None:
                self.connected_rsu.remove_vehicle(self.vehicle)

            self.connected_rsu = best_rsu

            if best_rsu is not None:
                best_rsu.register_vehicle(self.vehicle)

    # -----------------------------------------------------
    # IEEE 802.11p TRANSMISSION
    # -----------------------------------------------------
    def transmit(self):
        
        if self.connected_rsu is None:
            return

        now = time.time()

        # enforce 1–10 messages/sec
        if now - self.last_tx_time < self.tx_interval:
            return

        telemetry = self.get_telemetry()

        if telemetry is None:
            return

        payload = json.dumps(telemetry).encode()

        frame = IEEE80211pFrame(
            receiver_mac=self.connected_rsu.mac,
            sender_mac=self.vehicle.mac,
            payload=payload,
            sequence=int(now * 1000) % 4096
        )

        raw_frame = frame.build()

        # Simulated DSRC latency (10–100 ms)
        # latency = random.uniform(0.01, 0.1)
        # time.sleep(latency)
        # Simulated DSRC latency (0–5 ms)
        latency = random.uniform(0.0, 0.005)
        time.sleep(latency)
        self.packets_sent += 1
        # Deliver frame to RSU

      
        import metrics
        metrics.global_packets_sent += 1

        self.connected_rsu.receive_frame(self.vehicle, raw_frame)

        self.last_tx_time = now

    # -----------------------------------------------------
    # LONG WAIT ALERT
    # -----------------------------------------------------
    def send_long_wait_notification(self, wait_duration):

        if self.connected_rsu is None:
            return

        self.connected_rsu.receive_long_wait_notification(
            self.vehicle,
            wait_duration
        )


# ---------------------------------------------------------
# VEHICLE
# ---------------------------------------------------------
class Vehicle:

    def __init__(self, vid, network, start, destination):

        self.id = vid
        self.network = network
        self.current_node = start
        self.destination = destination

        self.target_node = None
        self.progress = 0

        self.mac = generate_mac()

        self.base_speed = random.uniform(0.002, 0.004)

        self.current_speed = self.base_speed
        self.prev_speed = self.base_speed

        self.color = (0, 0, 255)

        self.obu = OBU(self)

        self.wait_time = 0
        self.move_time = 0
        self.total_wait_time = 0

        self.next_notify_threshold = 500
        self.max_notify_threshold = 2400

        self.lane_id = random.randint(0, 1)

    # -----------------------------------------------------
    # UPDATE
    # -----------------------------------------------------
    def update(self, signals, vehicles, rsus=None, globally_congested=None):

        rsus = rsus or []

        if self.target_node is None:

            if self.current_node == self.destination:
                return

            rsu = None

            if rsus:
                rsu = next((r for r in rsus if r.node == self.current_node), None)

            if rsu:

                self.target_node = rsu.get_next_hop(
                    self.destination,
                    globally_congested
                )

            else:

                try:
                    path = self.network.dynamic_shortest_path(
                        self.current_node,
                        self.destination,
                        globally_congested
                    )

                    if len(path) > 1:
                        self.target_node = path[1]

                except:
                    pass

        if self.target_node is None:
            return

        speed = self.base_speed
        self.prev_speed = self.current_speed

        for signal in signals:

            if signal.node == self.target_node:

                lane_state = signal.state.get(
                    (self.current_node, self.target_node),
                    "GREEN"
                )

                if lane_state in ["RED", "YELLOW"]:

                    start_pos = self.network.positions[self.current_node]
                    end_pos = self.network.positions[self.target_node]

                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]

                    length = math.hypot(dx, dy)

                    if length == 0:
                        continue

                    stop_distance = NODE_RADIUS + 10
                    stop_progress = 1 - (stop_distance / length)

                    if self.progress >= stop_progress:
                        speed = 0

        for other in vehicles:

            if other is self:
                continue

            if (
                other.current_node == self.current_node
                and other.target_node == self.target_node
                and other.lane_id == self.lane_id
            ):

                if other.progress > self.progress:

                    gap = other.progress - self.progress

                    if gap < 0.03:
                        speed = 0

        if speed < 0.001:

            self.wait_time += 1
            self.total_wait_time += 1
            self.move_time = 0

        else:

            self.move_time += 1

            if self.move_time > 240:

                self.wait_time = 0
                self.next_notify_threshold = 200

        if self.wait_time >= self.next_notify_threshold:

            self.obu.send_long_wait_notification(self.wait_time)

            self.next_notify_threshold = min(
                int(self.next_notify_threshold * 1.5),
                self.max_notify_threshold
            )

        self.current_speed = speed

        if rsus:

            self.obu.handover(rsus)
            self.obu.transmit()

        self.progress += speed

        if self.progress >= 1:

            self.current_node = self.target_node
            self.progress = 0

            if self.current_node == self.destination:

                self.target_node = None
                return

            rsu = next((r for r in rsus if r.node == self.current_node), None)

            if rsu:

                self.target_node = rsu.get_next_hop(
                    self.destination,
                    globally_congested
                )

            else:

                try:

                    path = self.network.dynamic_shortest_path(
                        self.current_node,
                        self.destination,
                        globally_congested
                    )

                    if len(path) > 1:
                        self.target_node = path[1]

                except:

                    self.target_node = None

            if self.target_node:

                self.network.increase_traffic(
                    self.current_node,
                    self.target_node
                )

    # -----------------------------------------------------
    # DRAW
    # -----------------------------------------------------
    def draw(self, screen, zoom, offset_x, offset_y):

        if self.target_node is None:
            return

        start_pos = self.network.positions[self.current_node]
        end_pos = self.network.positions[self.target_node]

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        length = math.hypot(dx, dy)

        if length == 0:
            return

        # lane_offset_x = -dy / length * 5
        # lane_offset_y = dx / length * 5
        # Lane offset (0 -> center-left, 1 -> center-right)
        lane_width = 8
        offset_multiplier = -2.5 if self.lane_id == 0 else 2.5
        
        lane_offset_x = -dy / length * offset_multiplier
        lane_offset_y = dx / length * offset_multiplier

        x = start_pos[0] + dx * self.progress + lane_offset_x
        y = start_pos[1] + dy * self.progress + lane_offset_y

        screen_x = x * zoom + offset_x
        screen_y = y * zoom + offset_y

        angle = math.atan2(dy, dx)

        size = 4 * zoom

        front = (
            screen_x + math.cos(angle) * size,
            screen_y + math.sin(angle) * size
        )

        left = (
            screen_x + math.cos(angle + 2.5) * size,
            screen_y + math.sin(angle + 2.5) * size
        )

        right = (
            screen_x + math.cos(angle - 2.5) * size,
            screen_y + math.sin(angle - 2.5) * size
        )

        if self.obu.connected_rsu:
            color = (0, 200, 0)
        else:
            color = (200, 0, 0)

        pygame.draw.polygon(screen, color, [front, left, right])