# Block 20 — Simplified general calculator

## What changed

- Removed the separate `Golem quick capture` page.
- Golem is now selected the same way as Prospector and MOLE: via `Build profile` in the sidebar.
- The Golem build file is now easier to find: `configs/builds/golem_manual.yaml`.
- Removed visible `Operator / pilot`, `Crew size`, and `Run tag` controls from the sidebar.
- Resource/yield fields remain in the calculator because they are useful for future profit/yield analytics.

## Why

One general calculator is simpler for real collection:

```text
select build → enter rock → enter resource/yield → save event
```

Crew/operator metadata is still supported internally for old events and future use, but it is no longer required in the UI.

## MLOps value

The event schema stays stable, but the capture UX is simpler. This reduces manual-entry friction and should make 100+ real labeled runs easier to collect.
