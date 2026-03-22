import argparse
from flask import Flask, request, jsonify

from rsu import RSU
from rsu_client import rsu_network_client as rnc
from network import RoadNetwork


# ------------------------------------------------
# Build same road network used in simulator
# ------------------------------------------------
def build_network():

    net = RoadNetwork()

    net.add_node("A",(200,2800))
    net.add_node("B",(1000,2800))
    net.add_node("C",(200,2400))
    net.add_node("D",(1000,2400))
    net.add_node("E",(200,2000))
    net.add_node("F",(1000,2000))
    net.add_node("M1",(600,2000))
    net.add_node("G",(200,1600))
    net.add_node("H",(1000,1600))
    net.add_node("M2",(600,1600))
    net.add_node("I",(200,1200))
    net.add_node("J",(1000,800))
    net.add_node("K",(1000,400))
    net.add_node("L",(1800,800))

    net.add_road("B","D",bidirectional=True)
    net.add_road("A","C",bidirectional=True)
    net.add_road("C","D",bidirectional=True)
    net.add_road("C","E",bidirectional=True)
    net.add_road("D","F",bidirectional=True)
    net.add_road("E","G",bidirectional=True)
    net.add_road("F","H",bidirectional=True)
    net.add_road("I","J",bidirectional=True)
    net.add_road("I","G",bidirectional=True)
    net.add_road("G","M2",bidirectional=True)
    net.add_road("F","M1",bidirectional=True)
    net.add_road("M1","M2",bidirectional=True)
    net.add_road("H","M2",bidirectional=True)
    net.add_road("E","M1",bidirectional=True)
    net.add_road("M2","J",bidirectional=True)
    net.add_road("J","K",bidirectional=True)
    net.add_road("J","L",bidirectional=True)
    net.add_road("H","J",bidirectional=True)

    return net


# ------------------------------------------------
# Create RSU service
# ------------------------------------------------
def create_rsu_service(node, port, server_url):

    app = Flask(node)

    # build road network
    network = build_network()

    # create RSU
    rsu = RSU(node, network)

    # connect to V2X central server
    rnc.connect(server_url)

    print(f"[RSU SERVICE] RSU {node} running on port {port}")

    # ---------------------------------------------
    # Receive vehicle telemetry
    # ---------------------------------------------
    @app.route("/telemetry", methods=["POST"])
    def telemetry():

        data = request.json

        vehicle_id = data["vehicle_id"]

        # store telemetry
        rsu.receive_telemetry(vehicle_id, data)

        # check congestion via long wait aggregation
        rsu.check_long_wait_batch(client=rnc)

        return jsonify({"status": "ok"})


    # ---------------------------------------------
    # Receive long wait notifications
    # ---------------------------------------------
    @app.route("/long_wait", methods=["POST"])
    def long_wait():

        data = request.json

        vehicle_id = data["vehicle_id"]
        wait_time  = data["wait_time"]

        rsu.long_wait_buffer[vehicle_id] = wait_time

        rsu.check_long_wait_batch(client=rnc)

        return jsonify({"status": "ok"})


    # ---------------------------------------------
    # Get next hop routing instruction
    # ---------------------------------------------
    @app.route("/next_hop")
    def next_hop():

        destination = request.args.get("destination")

        # ask central server for congestion map
        globally_congested = rnc.get_active_congested_nodes()

        hop = rsu.get_next_hop(destination, globally_congested)

        return jsonify({"next_hop": hop})


    # start flask service
    app.run(host="0.0.0.0", port=port)


# ------------------------------------------------
# Entry point
# ------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--node", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server", default="192.168.137.6:5000")

    args = parser.parse_args()

    create_rsu_service(args.node, args.port, args.server)