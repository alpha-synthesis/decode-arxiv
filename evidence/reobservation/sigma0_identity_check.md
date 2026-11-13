# Identity check of the re-observation Monte Carlo at sigma = 0

Date: 2026-09-01.

Scope: this check certifies the arithmetic self-consistency of the Monte Carlo
executor against the dynamic program, both of which read the same dynamics
from the passed configuration, and that re-planning is idempotent at
sigma = 0. It certifies nothing about rotation worlds (the re-observation code
is linear only) and nothing about calibration at sigma > 0, which the
brute-force check of the Kalman belief covers.

Command:

    python -m benchmark.v8_reobservation \
      --world <pilot world> \
      --config <BASE configuration, process_noise 0> \
      --loans 150 --gate-q0

Output:

    GATE q0: |MC - DP| max 5.46e-12  |CL_oracle - OL_oracle| max 0.00e+00  PASS

The cohort came from a rotation-generated pilot world while the configuration
was the linear BASE configuration. Both sides of every compared quantity use
the same configuration-derived A, B and C, so the identity holds for what it
claims and no more.
