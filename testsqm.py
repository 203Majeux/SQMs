import socket
import time

#IP = "192.168.1.160"
IP = "192.168.1.167"
PORT = 10001

COMMANDS = [
    b"ix",       # Unit information
    b"cx",       # Calibration information
    b"rx",       # Averaged reading
    b"ux",       # Most recent unaveraged reading
]

for command in COMMANDS:
    print(f"\nTesting {command!r}")

    try:
        with socket.create_connection((IP, PORT), timeout=5) as sock:
            sock.settimeout(5)

            print("  Connected")
            time.sleep(0.5)

            sock.sendall(command)
            print(f"  Sent: {command!r}")

            try:
                response = sock.recv(2048)

                if response:
                    print(f"  Received {len(response)} bytes")
                    print(f"  Raw: {response!r}")
                    print(
                        "  Text:",
                        response.decode("ascii", errors="replace")
                    )
                else:
                    print("  Connection closed by the device")

            except socket.timeout:
                print("  TIMEOUT: no bytes returned")

    except OSError as exc:
        print(f"  CONNECTION ERROR: {exc}")

    time.sleep(1)