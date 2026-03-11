#!/bin/bash
# Check N300 machine setup for inference readiness

echo "=== IOMMU ==="
dmesg | grep -i iommu | head -5
echo "(want: no output, or 'disabled')"

echo ""
echo "=== Hugepages ==="
cat /proc/meminfo | grep Huge
echo "(want: HugePages_Total >= 1280)"

echo ""
echo "=== Hugepages mount ==="
mount | grep huge
echo "(want: hugetlbfs mounted at /dev/hugepages)"

echo ""
echo "=== Tenstorrent device ==="
ls -la /dev/tenstorrent/ 2>/dev/null || echo "No /dev/tenstorrent found"
echo "(want: crw-rw-rw- permissions)"

echo ""
echo "=== KMD version ==="
tt-smi --version 2>/dev/null || echo "tt-smi not found"

echo ""
echo "=== Firmware version ==="
tt-smi -s --snapshot_no_tty 2>/dev/null | grep -i firmware || echo "Could not read firmware info"

echo ""
echo "=== CPUs ==="
lscpu | grep -E "^CPU\(s\)|^Thread|^Core|^Socket|NUMA"
