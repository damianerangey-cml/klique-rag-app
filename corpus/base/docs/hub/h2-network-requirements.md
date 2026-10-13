# Hub H2 network requirements

## Connectivity

The Hub H2 supports Ethernet and 2.4 GHz Wi-Fi. It does not support 5 GHz Wi-Fi networks. A DHCP address is required unless a static address is configured in the mobile app.

## Firewall

The hub makes outbound connections only. Allow outbound TCP port 8883 for sensor data and outbound TCP port 443 for firmware downloads. No inbound ports need to be opened.

## Bandwidth

A hub with 50 sensors uses about 40 MB of data per month.
