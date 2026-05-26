# Block 32 — Resource profiles + sequential capture UI

Adds `configs/resources.yaml`, resource behavior fields, and a more linear UI
flow for real data capture.

Capture order:
1. Scan values.
2. Beam states.
3. Formula result / helper.
4. Composition / resources.
5. Outcome / observed behavior.
6. Refinery / future yield.
7. Save event.

Resource rows now support:
- resource_name
- resource_percent
- raw_scu_estimate
- observed_window_size
- observed_charge_behavior
- comment

Resource profile hints are not hard rules. They are capture/analytics hints;
real tuning should come from actual outcomes and calibration observations.
