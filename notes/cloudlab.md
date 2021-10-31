# Cloudlab

Cloudlab is a national scientific computing meta-cloud.

## Useful Nodes

The following nodes balance utility and availability.

|      Name |  Locale | Type | Nodes | CPUs | Cores | RAM | SSD | SSD Type |  HDD | Passmark |    Perf |
|-----------|---------|------|-------|------|-------|-----|-----|----------|------|----------|---------|
|     xl170 |    Utah |  CPU |   200 |    1 |    10 |  64 | 480 |     SATA |    0 |    11732 |     Low |
|      m510 |    Utah |  CPU |   270 |    1 |     8 |  64 | 256 |     NVMe |    0 |    11732 |     Low |
| c6525-25g |    Utah |  CPU |   144 |    1 |    16 | 128 | 960 |     NVMe |    0 |    32480 |    High |
|    c220g5 |    Wisc |  CPU |   224 |    2 |    20 | 192 | 480 |     SATA | 1000 |    21378 |  Medium |
|    c240g5 |    Wisc |  GPU |    32 |    2 |    20 | 192 | 480 |     SATA | 1000 |    21378 |  Medium |
|     c6420 | Clemson |  CPU |    72 |    2 |    32 | 384 |   0 |      N/A | 2000 |      N/A |    High |
|     r7525 | Clemson |  GPU |    15 |    2 |    64 | 512 |   0 |      N/A | 2000 |    72534 | Highest |

- Ubuntu 20.04 URN
  - `urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU14-64-STD`
