# UDP Relay

This utility forwards the flight simulator's local UDP traffic from a Windows
PC to a remote MacBook running the Python pilot. It is useful when the simulator
does not provide a setting for the destination IP address.

The relay forwards:

| Traffic | Port | Direction |
| --- | --- | --- |
| MAVLink telemetry | `14550` | Windows simulator to MacBook |
| MAVLink commands | `14550` | MacBook to Windows simulator |
| Vision frames | `5600` | Windows simulator to MacBook |

## Security

Use Tailscale or another private VPN. Do not expose UDP ports `14550` or `5600`
with router port forwarding.

The example MAVLink and vision protocols do not authenticate or encrypt packets
on their own. Tailscale provides encrypted device-to-device transport and lets
the relay work while the MacBook changes networks.

## 1. Install Tailscale

Install [Tailscale](https://tailscale.com/download) on the Windows simulator PC
and the MacBook. Sign both devices into the same Tailscale account.

On the MacBook, find its Tailscale IPv4 address:

```bash
tailscale ip -4
```

The address will usually look like `100.x.y.z`.

## 2. Configure the Relay

Open `udp_relay.py` on the Windows simulator PC and replace the placeholder with
the MacBook's Tailscale IPv4 address:

```python
MAC_IP = "100.x.y.z"
```

The default ports match the Python pilot example:

```python
MAVLINK_PORT = 14550
VISION_PORT = 5600
```

## 3. Configure the MacBook Pilot

In `Path_Optimizer/examples/pypilot/main.py`, listen for MAVLink traffic on all
local interfaces:

```python
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 14550
```

In `Path_Optimizer/examples/pypilot/vision_rx.py`, listen for vision frames on
all local interfaces:

```python
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600
```

Using `0.0.0.0` allows the pilot to receive packets through the MacBook's
Tailscale interface.

## 4. Configure the Windows Firewall

Allow inbound MAVLink commands on the Windows simulator PC only from the
MacBook's Tailscale IP address:

```powershell
New-NetFirewallRule `
  -DisplayName "Drone Sim MAVLink via Tailscale" `
  -Direction Inbound `
  -Protocol UDP `
  -LocalPort 14550 `
  -RemoteAddress 100.x.y.z `
  -Action Allow
```

Replace `100.x.y.z` with the MacBook's Tailscale IP address.

The vision stream normally flows outbound from Windows, so an inbound Windows
firewall rule for port `5600` is usually unnecessary.

## 5. Install the MacBook Pilot Dependencies

From the repository root on the MacBook:

```bash
cd Path_Optimizer/examples/pypilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Start the Simulator and Pilot

Start the components in this order:

1. Connect both computers to Tailscale.
2. On Windows, start the relay from the repository root:

   ```powershell
   python Path_Optimizer\src\udp_relay\udp_relay.py
   ```

3. Launch the flight simulator on Windows.
4. On the MacBook, run the pilot:

   ```bash
   cd Path_Optimizer/examples/pypilot
   source .venv/bin/activate
   python main.py
   ```

The relay prints status messages when it detects the simulator's MAVLink source,
receives the first MacBook command, and forwards the first vision packet.

The MacBook pilot should advance from:

```text
Waiting for heartbeat...
```

to:

```text
Connected to system: ...
```

## 7. Verify Network Traffic

If the pilot remains stuck waiting for a heartbeat, confirm that `MAC_IP` in
`udp_relay.py` is the MacBook's Tailscale IP address.

On the MacBook, inspect incoming packets:

```bash
sudo tcpdump -n -i any 'udp port 14550 or udp port 5600'
```

## Troubleshooting

### Ignored UDP port-unreachable response

Windows may report `ConnectionResetError: [WinError 10054]` when the relay sends
a UDP packet before a listener is ready on the MacBook. The relay catches this
condition, prints at most three warnings, and then suppresses further warnings.

Start the MacBook pilot before launching the simulator when practical.

### Telemetry works but commands do not

The relay learns the simulator's MAVLink return address from incoming telemetry.
If the simulator uses a separate hard-coded command port, update the relay to
send commands to that explicit port.

### Telemetry works but vision does not

Check port `5600` separately with `tcpdump`. Vision traffic is one-way from the
Windows simulator PC to the MacBook.
