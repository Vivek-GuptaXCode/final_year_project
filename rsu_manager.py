import multiprocessing
from rsu_service import create_rsu_service
import os
from dotenv import load_dotenv
load_dotenv()   


SERVER_URL = os.getenv("SERVER_URL")

# RSU node → port mapping
RSU_PORTS = {
    "C":5001,
    "D":5002,
    "E":5003,
    "F":5004,
    "G":5005,
    "H":5006,
    "J":5007,
    "M1":5008,
    "M2":5009,
}

def start_rsu(node, port):
    create_rsu_service(node, port, SERVER_URL)


if __name__ == "__main__":

    processes = []

    for node, port in RSU_PORTS.items():

        p = multiprocessing.Process(
            target=start_rsu,
            args=(node, port)
        )

        p.start()
        processes.append(p)

        print(f"Started RSU {node} on port {port}")

    for p in processes:
        p.join()