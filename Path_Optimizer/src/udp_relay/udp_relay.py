"""Forward local simulator UDP traffic to a remote pilot over a private network."""

import select
import socket

# Set this to the MacBook's Tailscale IP address.
MAC_IP = "100.x.y.z"
MAVLINK_PORT = 14550
VISION_PORT = 5600
MAX_PORT_UNREACHABLE_WARNINGS = 3

port_unreachable_count = 0


def create_socket(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))

    # Prevent Windows from surfacing ICMP port-unreachable replies as fatal
    # ConnectionResetError exceptions for this UDP relay.
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)

    return sock


def receive(sock):
    global port_unreachable_count

    try:
        return sock.recvfrom(65536)
    except ConnectionResetError:
        port_unreachable_count += 1
        if port_unreachable_count <= MAX_PORT_UNREACHABLE_WARNINGS:
            print(
                "Ignored UDP port-unreachable response "
                f"({port_unreachable_count}/{MAX_PORT_UNREACHABLE_WARNINGS})",
                flush=True,
            )
        elif port_unreachable_count == MAX_PORT_UNREACHABLE_WARNINGS + 1:
            print("Further UDP port-unreachable warnings will be suppressed.", flush=True)
        return None


mav = create_socket(MAVLINK_PORT)
vision = create_socket(VISION_PORT)
latest_sim_mavlink_address = None
saw_mac_command = False
saw_early_mac_command = False
saw_vision_packet = False

print(f"UDP relay ready. Forwarding simulator traffic to {MAC_IP}.", flush=True)
print(f"Listening for local MAVLink telemetry on UDP {MAVLINK_PORT}.", flush=True)
print(f"Listening for local vision frames on UDP {VISION_PORT}.", flush=True)

while True:
    readable, _, _ = select.select([mav, vision], [], [])

    for sock in readable:
        received = receive(sock)
        if received is None:
            continue

        packet, sender = received

        if sock is mav:
            if sender[0] == MAC_IP:
                if latest_sim_mavlink_address:
                    mav.sendto(packet, latest_sim_mavlink_address)
                    if not saw_mac_command:
                        print(
                            f"Received first MAVLink command from MacBook at {sender[0]}.",
                            flush=True,
                        )
                        saw_mac_command = True
                elif not saw_early_mac_command:
                    print(
                        "Received a MacBook command before simulator telemetry; "
                        "waiting to learn the simulator address.",
                        flush=True,
                    )
                    saw_early_mac_command = True
            else:
                if latest_sim_mavlink_address != sender:
                    latest_sim_mavlink_address = sender
                    print(
                        "Simulator MAVLink source detected at "
                        f"{sender[0]}:{sender[1]}; forwarding telemetry to "
                        f"{MAC_IP}:{MAVLINK_PORT}.",
                        flush=True,
                    )
                mav.sendto(packet, (MAC_IP, MAVLINK_PORT))

        elif sender[0] != MAC_IP:
            vision.sendto(packet, (MAC_IP, VISION_PORT))
            if not saw_vision_packet:
                print(
                    "Received first vision packet; forwarding frames to "
                    f"{MAC_IP}:{VISION_PORT}.",
                    flush=True,
                )
                saw_vision_packet = True
