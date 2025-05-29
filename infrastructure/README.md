# HFT Bot Infrastructure Setup Guide

This guide provides detailed instructions for setting up the optimal infrastructure for the Hybrid Retail HFT Trading Bot.

## 🏗️ Infrastructure Overview

The HFT bot requires a carefully optimized infrastructure to achieve the target latency of <50ms execution times. This includes:

1. **High-Performance VPS** - Geographically optimized server
2. **Network Optimization** - Ultra-low latency connectivity
3. **Hardware Optimization** - CPU, RAM, and storage tuning
4. **Software Optimization** - OS and application-level tuning

## 🌐 VPS Selection and Setup

### Recommended VPS Providers

#### 1. Equinix Metal (Recommended)
- **Location**: NY4 (New York), LD4 (London), TY11 (Tokyo)
- **Hardware**: AMD EPYC 7003 series or Intel Xeon Ice Lake
- **Network**: 10Gbps+ with cross-connects to major brokers
- **Latency**: <1ms to major financial exchanges

```bash
# Example Equinix Metal configuration
Server Type: c3.medium.x86
CPU: AMD EPYC 7402P (24 cores, 2.8GHz base, 3.35GHz boost)
RAM: 64GB DDR4-3200
Storage: 2x 480GB NVMe SSD
Network: 2x 10Gbps
```

#### 2. Vultr High Frequency
- **Location**: New York, London, Tokyo
- **Hardware**: Intel Xeon E-2288G
- **Network**: 10Gbps
- **Latency**: <5ms to major brokers

#### 3. Beeks FX VPS (Forex-Specific)
- **Location**: LD4, NY4, TY3
- **Specialization**: Forex trading optimization
- **Network**: Direct broker connections
- **Latency**: <1ms to major forex brokers

### VPS Configuration Script

```bash
#!/bin/bash
# VPS Optimization Script for HFT Bot

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    build-essential \
    cmake \
    git \
    htop \
    iotop \
    net-tools \
    python3.11 \
    python3.11-dev \
    python3-pip \
    redis-server \
    sqlite3 \
    tmux \
    vim \
    wget \
    curl

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# System optimization
echo "Applying system optimizations..."

# CPU optimization
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Network optimization
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sudo sysctl -w net.core.netdev_max_backlog=5000

# Memory optimization
sudo sysctl -w vm.swappiness=1
sudo sysctl -w vm.dirty_ratio=15
sudo sysctl -w vm.dirty_background_ratio=5

# Make optimizations persistent
sudo tee -a /etc/sysctl.conf << EOF
# HFT Bot Optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
EOF

echo "VPS optimization complete!"
```

## 🔧 Hardware Specifications

### Minimum Requirements
- **CPU**: 4 cores, 3.0GHz+
- **RAM**: 16GB DDR4
- **Storage**: 250GB NVMe SSD
- **Network**: 1Gbps

### Recommended Specifications
- **CPU**: 8+ cores, 4.0GHz+ (AMD Ryzen 9 5950X or Intel i9-12900K)
- **RAM**: 32GB DDR4-3200 or DDR5
- **Storage**: 500GB+ NVMe SSD (Samsung 980 PRO or similar)
- **Network**: 10Gbps with low-latency NIC

### Optimal Specifications
- **CPU**: 16+ cores, 4.5GHz+ (AMD Ryzen 9 7950X or Intel i9-13900K)
- **RAM**: 64GB DDR5-5600
- **Storage**: 1TB+ NVMe SSD RAID 0
- **Network**: 25Gbps+ with DPDK-capable NIC

## 🌍 Geographic Optimization

### Broker Server Locations

#### Major Forex Brokers
- **IC Markets**: NY4 (Equinix New York)
- **Pepperstone**: LD4 (Equinix London)
- **FXCM**: NY4 (Equinix New York)
- **Oanda**: NY4, LD4, TY11

#### Recommended VPS Locations
1. **New York (NY4)** - For US/Americas trading
2. **London (LD4)** - For European trading
3. **Tokyo (TY11)** - For Asian trading

### Latency Testing Script

```bash
#!/bin/bash
# Broker Latency Testing Script

BROKERS=(
    "icmarkets-demo.com"
    "pepperstone-demo.com"
    "fxcm-demo.com"
    "oanda-demo.com"
)

echo "Testing latency to major brokers..."
for broker in "${BROKERS[@]}"; do
    echo "Testing $broker:"
    ping -c 10 $broker | tail -1 | awk '{print $4}' | cut -d '/' -f 2
    echo ""
done
```

## 🔌 Network Optimization

### Network Interface Configuration

```bash
#!/bin/bash
# Network Interface Optimization

# Find network interface
INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)

# Optimize network interface
sudo ethtool -G $INTERFACE rx 4096 tx 4096
sudo ethtool -K $INTERFACE gro off
sudo ethtool -K $INTERFACE lro off
sudo ethtool -C $INTERFACE rx-usecs 0 tx-usecs 0

# Set CPU affinity for network interrupts
echo "Setting CPU affinity for network interrupts..."
for irq in $(grep $INTERFACE /proc/interrupts | cut -d: -f1); do
    echo 2 | sudo tee /proc/irq/$irq/smp_affinity
done
```

### Firewall Configuration

```bash
#!/bin/bash
# Firewall Configuration for HFT Bot

# Reset iptables
sudo iptables -F
sudo iptables -X
sudo iptables -t nat -F
sudo iptables -t nat -X

# Default policies
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT

# Allow loopback
sudo iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
sudo iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (change port as needed)
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HFT Bot ports
sudo iptables -A INPUT -p tcp --dport 5555:5557 -j ACCEPT  # ZMQ ports
sudo iptables -A INPUT -p tcp --dport 12000 -j ACCEPT      # Dashboard

# Allow MT5 ports
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT        # HTTPS
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT         # HTTP

# Save rules
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

## 🐧 Operating System Optimization

### Ubuntu 22.04 LTS Optimization

```bash
#!/bin/bash
# Ubuntu Optimization for HFT

# Disable unnecessary services
sudo systemctl disable snapd
sudo systemctl disable bluetooth
sudo systemctl disable cups
sudo systemctl disable avahi-daemon

# Install real-time kernel (optional)
sudo apt install -y linux-lowlatency

# Configure CPU governor
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils

# Disable CPU frequency scaling
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done

# Configure huge pages
echo 'vm.nr_hugepages = 1024' | sudo tee -a /etc/sysctl.conf

# Disable swap
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# Set process priorities
echo '@hftbot soft rtprio 99' | sudo tee -a /etc/security/limits.conf
echo '@hftbot hard rtprio 99' | sudo tee -a /etc/security/limits.conf
```

## 🐳 Docker Deployment (Optional)

### Dockerfile

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 5555 5556 5557 12000

# Run application
CMD ["python3", "main.py", "--mode", "all"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  hft-bot:
    build: .
    ports:
      - "5555:5555"
      - "5556:5556"
      - "5557:5557"
      - "12000:12000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
```

## 📊 Performance Monitoring

### System Monitoring Script

```bash
#!/bin/bash
# System Performance Monitoring

# Create monitoring script
cat > /usr/local/bin/hft-monitor.sh << 'EOF'
#!/bin/bash

LOG_FILE="/var/log/hft-performance.log"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Memory usage
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    # Network latency to broker
    LATENCY=$(ping -c 1 icmarkets-demo.com | tail -1 | awk '{print $4}' | cut -d '/' -f 2)
    
    # Disk I/O
    DISK_IO=$(iostat -d 1 1 | tail -n +4 | awk '{print $4}')
    
    echo "$TIMESTAMP,CPU:$CPU_USAGE%,MEM:$MEM_USAGE%,LAT:${LATENCY}ms,IO:$DISK_IO" >> $LOG_FILE
    
    sleep 10
done
EOF

chmod +x /usr/local/bin/hft-monitor.sh

# Create systemd service
cat > /etc/systemd/system/hft-monitor.service << EOF
[Unit]
Description=HFT Performance Monitor
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/hft-monitor.sh
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOF

systemctl enable hft-monitor.service
systemctl start hft-monitor.service
```

## 🔒 Security Configuration

### SSH Hardening

```bash
#!/bin/bash
# SSH Security Configuration

# Backup original config
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Configure SSH
sudo tee /etc/ssh/sshd_config << EOF
Port 2222
Protocol 2
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM yes
X11Forwarding no
PrintMotd no
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
ClientAliveInterval 300
ClientAliveCountMax 2
MaxAuthTries 3
MaxSessions 2
EOF

# Restart SSH
sudo systemctl restart sshd
```

### Fail2Ban Configuration

```bash
#!/bin/bash
# Install and configure Fail2Ban

sudo apt install -y fail2ban

# Configure Fail2Ban
sudo tee /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = 2222
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
EOF

sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] VPS provisioned in optimal location
- [ ] Operating system optimized
- [ ] Network configuration completed
- [ ] Security hardening applied
- [ ] Monitoring tools installed

### Deployment
- [ ] HFT bot code deployed
- [ ] Configuration files updated
- [ ] Dependencies installed
- [ ] Database initialized
- [ ] SSL certificates configured (if needed)

### Post-Deployment
- [ ] Latency testing completed
- [ ] Performance monitoring active
- [ ] Backup procedures in place
- [ ] Alert systems configured
- [ ] Documentation updated

### Performance Validation
- [ ] Execution latency < 50ms
- [ ] Network latency < 5ms to broker
- [ ] CPU usage < 80% under load
- [ ] Memory usage < 70%
- [ ] No packet loss detected

## 🚨 Troubleshooting

### Common Issues

#### High Latency
```bash
# Check network configuration
ping -c 10 broker-server.com
traceroute broker-server.com
netstat -i

# Check CPU frequency scaling
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check for CPU throttling
dmesg | grep -i throttl
```

#### Memory Issues
```bash
# Check memory usage
free -h
cat /proc/meminfo

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python3 main.py
```

#### Network Issues
```bash
# Check network interface statistics
cat /proc/net/dev
ethtool -S eth0

# Check for dropped packets
netstat -i
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
1. **Daily**: Monitor performance metrics
2. **Weekly**: Update system packages
3. **Monthly**: Review and optimize configuration
4. **Quarterly**: Hardware performance review

### Emergency Procedures
1. **High Latency**: Switch to backup VPS
2. **System Failure**: Activate disaster recovery
3. **Security Breach**: Isolate and investigate
4. **Data Loss**: Restore from backup

For additional support, refer to the main documentation or contact the development team.