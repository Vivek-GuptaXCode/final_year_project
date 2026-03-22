# import math
# import time
# import struct
# import zlib
# import json
# import random

# import metrics




# # -------------------------------------------------------
# # MAC generator
# # -------------------------------------------------------
# def generate_mac():
#     return "AA:%02x:%02x:%02x:%02x:%02x" % tuple(
#         random.randint(0, 255) for _ in range(5)
#     )


# class RSU:

#     def __init__(self, node, network, radius=100):
#         self.packets_received = 0
#         self.bytes_received = 0
#         self.latencies = []
#         self.node = node
#         self.network = network
#         self.radius = radius

#         # RSU MAC address (for IEEE 802.11p)
#         self.mac = generate_mac()

#         # ---- Alert cooldown ----
#         self.alert_cooldown = 5.0
#         self._last_alert_time = 0.0

#         # ---- V2I layer ----
#         self.registered_vehicles = {}   # vid → latest telemetry
#         self.vehicle_objects = {}       # vid → Vehicle object

#         # ---- Long Wait Aggregation ----
#         self.long_wait_buffer = {}      # vid → last_wait_duration
#         self.last_batch_alert_time = 0.0
#         self.batch_window = 10.0
#         self.batch_threshold = 12
#         self.is_congested = False

#         self.signal = None


#     # -------------------------------------------------------
#     # POSITION
#     # -------------------------------------------------------
#     def get_position(self):
#         return self.network.positions[self.node]


#     # -------------------------------------------------------
#     # VEHICLE REGISTRATION
#     # -------------------------------------------------------
#     def register_vehicle(self, vehicle):
#         self.vehicle_objects[vehicle.id] = vehicle


#     def remove_vehicle(self, vehicle):

#         self.registered_vehicles.pop(vehicle.id, None)
#         self.vehicle_objects.pop(vehicle.id, None)


#     # -------------------------------------------------------
#     # IEEE 802.11p FRAME RECEPTION
#     # -------------------------------------------------------
#     def receive_frame(self, vehicle, frame):

#         try:

#             # Minimum frame length
#             if len(frame) < 28:
#                 return

#             # Separate payload and FCS
#             payload = frame[24:-4]
#             received_fcs = struct.unpack("!I", frame[-4:])[0]

#             # Verify CRC
#             calc_fcs = zlib.crc32(frame[:-4]) & 0xffffffff

#             if received_fcs != calc_fcs:
#                 return

#             telemetry = json.loads(payload.decode())

#             self.receive_telemetry(vehicle, telemetry)

#         except Exception:
#             pass


#     # -------------------------------------------------------
#     # TELEMETRY STORAGE
#     # -------------------------------------------------------
#     def receive_telemetry(self, vehicle, telemetry):
#         self.packets_received += 1
     
#         import metrics
#         metrics.global_packets_received += 1
#         # estimate packet size
#         self.bytes_received += len(json.dumps(telemetry))

#         # latency calculation
#         if "timestamp" in telemetry:
#             latency = time.time() - telemetry["timestamp"]
#             self.latencies.append(latency)

#         # self.registered_vehicles[vehicle.id] = telemetry
#         vid = vehicle.id if hasattr(vehicle, 'id') else vehicle
#         self.registered_vehicles[vid] = telemetry


#     # -------------------------------------------------------
#     # LONG WAIT BUFFER
#     # -------------------------------------------------------
#     def receive_long_wait_notification(self, vehicle, wait_duration):

#         self.long_wait_buffer[vehicle.id] = wait_duration


#     # -------------------------------------------------------
#     # RSU ROUTING
#     # -------------------------------------------------------
#     def get_next_hop(self, destination, globally_congested=None):

#         if self.node == destination:
#             return None

#         try:

#             path = self.network.dynamic_shortest_path(
#                 self.node,
#                 destination,
#                 globally_congested=globally_congested
#             )

#             if len(path) > 1:
#                 return path[1]

#         except:
#             pass

#         return None


#     # -------------------------------------------------------
#     # CONGESTION DETECTION
#     # -------------------------------------------------------
#     def check_long_wait_batch(self, client=None):

#         if client is None or not client.is_connected:
#             return

#         now = time.time()

#         # ---- Sliding Window: Cleanup old entries first ----
#         # Remove any entries older than batch_window (15s)
#         self.long_wait_buffer = {
#             vid: val for vid, val in self.long_wait_buffer.items()
#             if now - val[1] <= self.batch_window
#         }

#         buffer_count = len(self.long_wait_buffer)

#         # ---------------------------------------------------
#         # 1. AGGREGATE ALERT
#         # ---------------------------------------------------
#         if buffer_count >= self.batch_threshold:

#             avg_wait = sum(v[0] for v in self.long_wait_buffer.values()) / buffer_count

#             client.send_junction_congestion_alert(
#                 self.node,
#                 buffer_count,
#                 round(avg_wait, 1)
#             )

#             # We don't clear the buffer here anymore, let sliding window handle it
#             # so that slightly staggered reports still count towards threshold
#             self.last_batch_alert_time = now
#             self.is_congested = True

#         # ---------------------------------------------------
#         # 2. CLEARANCE DETECTION
#         # ---------------------------------------------------
#         elif self.is_congested:

#             telemetry_list = list(self.registered_vehicles.values())
#             num_vehicles = len(telemetry_list)

#             if num_vehicles == 0:

#                 client.send_junction_clear_alert(self.node)

#                 self.is_congested = False
#                 self.long_wait_buffer.clear()

#                 return

#             current_queue = sum(
#                 1 for t in telemetry_list
#                 if t["speed"] < 0.001
#             )

#             avg_speed = sum(
#                 t["speed"] for t in telemetry_list
#             ) / num_vehicles

#             clearance_threshold = 5

#             if current_queue <= clearance_threshold and avg_speed > 0.005:
#                 client.send_junction_clear_alert(self.node)

#                 self.is_congested = False
#                 self.long_wait_buffer.clear()
#                 self.last_batch_alert_time = now
#         pass
    # def check_long_wait_batch(self, client=None):

    #     if client is None or not client.is_connected:
    #         return

    #     now = time.time()

    #     if now - self.last_batch_alert_time < 2.0:
    #         return

    #     telemetry_list = list(self.registered_vehicles.values())
    #     num_vehicles = len(telemetry_list)

    #     # ---------------------------------------------------
    #     # 0. SPEED BASED CLEARANCE (NEW)
    #     # ---------------------------------------------------
    #     if self.is_congested and num_vehicles > 0:

    #         avg_speed = sum(t["speed"] for t in telemetry_list) / num_vehicles

    #         # if vehicles are moving normally → clear congestion
    #         if avg_speed > 0.003:

    #             client.send_junction_clear_alert(self.node)

    #             self.is_congested = False
    #             self.long_wait_buffer.clear()
    #             self.last_batch_alert_time = now

    #             return

    #     buffer_count = len(self.long_wait_buffer)

    #     # ---------------------------------------------------
    #     # 1. AGGREGATE ALERT
    #     # ---------------------------------------------------
    #     if buffer_count >= self.batch_threshold:

    #         avg_wait = sum(self.long_wait_buffer.values()) / buffer_count

    #         client.send_junction_congestion_alert(
    #             self.node,
    #             buffer_count,
    #             round(avg_wait, 1)
    #         )

    #         self.long_wait_buffer.clear()
    #         self.last_batch_alert_time = now
    #         self.is_congested = True

    #     # ---------------------------------------------------
    #     # 2. QUEUE BASED CLEARANCE
    #     # ---------------------------------------------------
    #     elif self.is_congested:

    #         if num_vehicles == 0:

    #             client.send_junction_clear_alert(self.node)

    #             self.is_congested = False
    #             self.long_wait_buffer.clear()
    #             return

    #         current_queue = sum(
    #             1 for t in telemetry_list
    #             if t["speed"] < 0.001
    #         )

    #         avg_speed = sum(
    #             t["speed"] for t in telemetry_list
    #         ) / num_vehicles

    #         clearance_threshold = 3

    #         if current_queue <= clearance_threshold and avg_speed > 0.0025:

    #             client.send_junction_clear_alert(self.node)

    #             self.is_congested = False
    #             self.long_wait_buffer.clear()
    #             self.last_batch_alert_time = now

    #     # ---------------------------------------------------
    #     # BUFFER CLEANUP
    #     # ---------------------------------------------------
    #     if now - self.last_batch_alert_time > self.batch_window:

    #         self.long_wait_buffer.clear()
    #         self.last_batch_alert_time = now        


    # def check_long_wait_batch(self, client=None):


    #     if client is None or not client.is_connected:
    #         return

    #     now = time.time()

    #     # prevent sending alerts too frequently
    #     if now - self.last_batch_alert_time < 2.0:
    #         return

    #     buffer_count = len(self.long_wait_buffer)

    #     # ---------------------------------------------------
    #     # 1. AGGREGATE CONGESTION ALERT
    #     # ---------------------------------------------------
    #     if buffer_count >= self.batch_threshold:

    #         avg_wait = sum(self.long_wait_buffer.values()) / buffer_count

    #         client.send_junction_congestion_alert(
    #             self.node,
    #             buffer_count,
    #             round(avg_wait, 1)
    #         )

    #         self.long_wait_buffer.clear()
    #         self.last_batch_alert_time = now
    #         self.is_congested = True


    #     # ---------------------------------------------------
    #     # 2. REALISTIC CLEARANCE DETECTION
    #     # ---------------------------------------------------
    #     elif self.is_congested:

    #         telemetry_list = list(self.registered_vehicles.values())
    #         num_vehicles = len(telemetry_list)

    #         if num_vehicles == 0:

    #             client.send_junction_clear_alert(self.node)

    #             self.is_congested = False
    #             self.long_wait_buffer.clear()
    #             return

    #         # vehicles stopped (queue)
    #         # current_queue = sum(
    #         #     1 for t in telemetry_list
    #         #     if t["speed"] < 0.001
    #         # )
    #         # signal_wait_limit = 20  # seconds

    #         # current_queue = sum(
    #         #     1 for t in telemetry_list
    #         #     if t["speed"] < 0.001 and t.get("wait_time", 0) > signal_wait_limit
    #         # )
    #         # approximate one signal cycle (seconds)
    #         FPS = 40

    #         signal_cycle = 15 * FPS
    #         congestion_wait = signal_cycle * 2

    #         current_queue = sum(
    #             1 for t in telemetry_list
    #             if t["speed"] < 0.001 and t.get("wait_time", 0) > congestion_wait
    #         )
    #         # average speed
    #         avg_speed = sum(
    #             t["speed"] for t in telemetry_list
    #         ) / num_vehicles

    #         clearance_threshold = 2

    #         # clear congestion if traffic flows normally
    #         if current_queue <= clearance_threshold and avg_speed > 0.0025:

    #             client.send_junction_clear_alert(self.node)

    #             self.is_congested = False
    #             self.long_wait_buffer.clear()
    #             self.last_batch_alert_time = now


    #     # ---------------------------------------------------
    #     # 3. BUFFER CLEANUP
    #     # ---------------------------------------------------
    #     if now - self.last_batch_alert_time > self.batch_window:

    #         self.long_wait_buffer.clear()
    #         self.last_batch_alert_time = now

    # def check_long_wait_batch(self, client=None):

    #     if client is None or not client.is_connected:
    #         return

    #     now = time.time()

    #     if now - self.last_batch_alert_time < 2.0:
    #         return

    #     telemetry_list = list(self.registered_vehicles.values())
    #     num_vehicles = len(telemetry_list)

    #     if num_vehicles == 0:
    #         return

    #     # -------------------------------
    #     # Traffic statistics
    #     # -------------------------------

    #     avg_speed = sum(t["speed"] for t in telemetry_list) / num_vehicles

    #     stopped_vehicles = sum(
    #         1 for t in telemetry_list
    #         if t["speed"] < 0.001
    #     )

    #     # wait_time is measured in frames
    #     FPS = 40

    #     # one full signal cycle (red + green)
    #     signal_cycle = 15 * FPS

    #     # vehicles waiting longer than 2 cycles
    #     long_wait_queue = sum(
    #         1 for t in telemetry_list
    #         if t["speed"] < 0.001 and t.get("wait_time", 0) > signal_cycle * 2
    #     )

    #     # -------------------------------
    #     # CONGESTION DETECTION
    #     # -------------------------------

    #     if not self.is_congested:

    #         # congestion if many vehicles wait through multiple signals
    #         if long_wait_queue >= 5:

    #             client.send_junction_congestion_alert(
    #                 self.node,
    #                 long_wait_queue,
    #                 round(avg_speed, 4)
    #             )

    #             self.is_congested = True
    #             self.last_batch_alert_time = now

    #     # -------------------------------
    #     # CLEARANCE DETECTION
    #     # -------------------------------

    #     else:

    #         # if vehicles start moving normally → clear congestion
    #         if avg_speed > 0.0025 and stopped_vehicles <= 2:

    #             client.send_junction_clear_alert(self.node)

    #             self.is_congested = False
    #             self.last_batch_alert_time = now
    
    # def receive_frame(self, vehicle, frame):

    #     try:

    #         payload = frame[24:-4]
    #         telemetry = json.loads(payload.decode())

    #         self.receive_telemetry(vehicle, telemetry)

    #         print(f"[RSU {self.node}] telemetry received from {vehicle.id}")

    #     except Exception as e:
    #         print("Frame decode error:", e)



import math
import time
import struct
import zlib
import json
import random

import metrics




# -------------------------------------------------------
# MAC generator
# -------------------------------------------------------
def generate_mac():
    return "AA:%02x:%02x:%02x:%02x:%02x" % tuple(
        random.randint(0, 255) for _ in range(5)
    )


class RSU:

    def __init__(self, node, network, radius=120):
        self.packets_received = 0
        self.bytes_received = 0
        self.latencies = []
        self.node = node
        self.network = network
        self.radius = radius

        # RSU MAC address (for IEEE 802.11p)
        self.mac = generate_mac()

        # ---- Alert cooldown ----
        self.alert_cooldown = 5.0
        self._last_alert_time = 0.0

        # ---- V2I layer ----
        self.registered_vehicles = {}   # vid → latest telemetry
        self.vehicle_objects = {}       # vid → Vehicle object

        # ---- Long Wait Aggregation ----
        self.long_wait_buffer = {}      # vid → last_wait_duration
        self.last_batch_alert_time = 0.0
        self.batch_window = 15.0
        self.batch_threshold = 12.0
        self.is_congested = False

        self.signal = None


    # -------------------------------------------------------
    # POSITION
    # -------------------------------------------------------
    def get_position(self):
        return self.network.positions[self.node]


    # -------------------------------------------------------
    # VEHICLE REGISTRATION
    # -------------------------------------------------------
    def register_vehicle(self, vehicle):
        self.vehicle_objects[vehicle.id] = vehicle


    def remove_vehicle(self, vehicle):

        self.registered_vehicles.pop(vehicle.id, None)
        self.vehicle_objects.pop(vehicle.id, None)


    # -------------------------------------------------------
    # IEEE 802.11p FRAME RECEPTION
    # -------------------------------------------------------
    def receive_frame(self, vehicle, frame):

        try:

            # Minimum frame length
            if len(frame) < 28:
                return

            # Separate payload and FCS
            payload = frame[24:-4]
            received_fcs = struct.unpack("!I", frame[-4:])[0]

            # Verify CRC
            calc_fcs = zlib.crc32(frame[:-4]) & 0xffffffff

            if received_fcs != calc_fcs:
                return

            telemetry = json.loads(payload.decode())

            self.receive_telemetry(vehicle, telemetry)

        except Exception:
            pass


    # -------------------------------------------------------
    # TELEMETRY STORAGE
    # -------------------------------------------------------
    def receive_telemetry(self, vehicle, telemetry):
        self.packets_received += 1
     
        import metrics
        metrics.global_packets_received += 1
        # estimate packet size
        self.bytes_received += len(json.dumps(telemetry))

        # latency calculation
        if "timestamp" in telemetry:
            latency = time.time() - telemetry["timestamp"]
            self.latencies.append(latency)

        vid = vehicle.id if hasattr(vehicle, 'id') else vehicle
        self.registered_vehicles[vid] = telemetry


    # -------------------------------------------------------
    # LONG WAIT BUFFER
    # -------------------------------------------------------
    def receive_long_wait_notification(self, vehicle, wait_duration):
        # Store (wait_duration, capture_time)
        self.long_wait_buffer[vehicle.id] = (wait_duration, time.time())


    # -------------------------------------------------------
    # RSU ROUTING
    # -------------------------------------------------------
    def get_next_hop(self, destination, globally_congested=None):

        if self.node == destination:
            return None

        try:

            path = self.network.dynamic_shortest_path(
                self.node,
                destination,
                globally_congested=globally_congested
            )

            if len(path) > 1:
                return path[1]

        except:
            pass

        return None


    # -------------------------------------------------------
    # CONGESTION DETECTION
    # -------------------------------------------------------
    def check_long_wait_batch(self, client=None):

        if client is None or not client.is_connected:
            return

        now = time.time()

        # ---- Sliding Window: Cleanup old entries first ----
        # Remove any entries older than batch_window (15s)
        self.long_wait_buffer = {
            vid: val for vid, val in self.long_wait_buffer.items()
            if now - val[1] <= self.batch_window
        }

        buffer_count = len(self.long_wait_buffer)

        # ---------------------------------------------------
        # 1. AGGREGATE ALERT
        # ---------------------------------------------------
        if buffer_count >= self.batch_threshold:

            avg_wait = sum(v[0] for v in self.long_wait_buffer.values()) / buffer_count

            client.send_junction_congestion_alert(
                self.node,
                buffer_count,
                round(avg_wait, 1)
            )

            # We don't clear the buffer here anymore, let sliding window handle it
            # so that slightly staggered reports still count towards threshold
            self.last_batch_alert_time = now
            self.is_congested = True

        # ---------------------------------------------------
        # 2. CLEARANCE DETECTION
        # ---------------------------------------------------
        elif self.is_congested:

            telemetry_list = list(self.registered_vehicles.values())
            num_vehicles = len(telemetry_list)

            if num_vehicles == 0:

                client.send_junction_clear_alert(self.node)

                self.is_congested = False
                self.long_wait_buffer.clear()

                return

            current_queue = sum(
                1 for t in telemetry_list
                if t["speed"] < 0.001
            )

            avg_speed = sum(
                t["speed"] for t in telemetry_list
            ) / num_vehicles

            clearance_threshold = 5

            if current_queue <= clearance_threshold and avg_speed > 0.001:
                client.send_junction_clear_alert(self.node)

                self.is_congested = False
                self.long_wait_buffer.clear()
                self.last_batch_alert_time = now
        pass

    # def receive_frame(self, vehicle, frame):

    #     try:

    #         payload = frame[24:-4]
    #         telemetry = json.loads(payload.decode())

    #         self.receive_telemetry(vehicle, telemetry)

    #         print(f"[RSU {self.node}] telemetry received from {vehicle.id}")

    #     except Exception as e:
    #         print("Frame decode error:", e)