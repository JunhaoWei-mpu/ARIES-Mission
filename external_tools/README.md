# External solver installation

The binary is intentionally excluded from the public archive because it has
separate academic-use terms. Reproduce the exact-reference environment as
follows.

## Concorde 03.12.19 with QSopt

- Official page: https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm
- Linux binary: `codes/linux24/concorde.gz`
- Expected SHA-256 after decompression:
  `38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0`
- Install as `external_tools/bin/concorde` and make it executable.

The official 32-bit static executable may require execution outside restrictive
seccomp sandboxes.  Its legacy small-instance routine also rejects long edges
when a problem has fewer than ten nodes.  The wrapper therefore pads such
instances with zero-cost duplicates of Home, removes those duplicates from the
returned cycle, and verifies the reported integer objective against the
unpadded matrix before accepting a certificate.
