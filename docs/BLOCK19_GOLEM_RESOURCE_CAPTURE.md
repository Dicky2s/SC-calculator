# Block 19 — Golem + resource/yield capture

## What changed

This block prepares the app for real duo mining data collection before a 100+ run dataset is collected.

### 1. General calculator tab

A dedicated Streamlit tab was added:

```text
General calculator
```

It uses:

```text
configs/builds/golem_manual.yaml
```

The page writes events with:

```text
source = golem_quick_page
ship_type = golem
build_id = golem_pitman_rieger_stampede_v1
```

This makes it easier for a second operator to enter Golem runs quickly without changing the generic build selector.

### 2. Golem/Pitman config

Added:

```text
pitman_s1
```

to `configs/heads.yaml` and a Golem build profile with a Pitman mining laser.

### 3. All known ship mining modules

`configs/modules.yaml` now contains the known ship mining module set:

```text
Brandt, FLTR, FLTR-L, FLTR-XL, Focus, Focus II, Focus III, Forel, Lifeline,
Optimum, Rieger, Rieger-C2, Rieger-C3, Rime, ROC, Stampede, Surge, Torpid,
Torrent, Torrent II, Torrent III, Vaux, Vaux-C2, Vaux-C3, XTR, XTR-L, XTR-XL
```

The coefficients are still local approximation/calibration values. They are not the authoritative Star Citizen formula.

### 4. Active module selection

The calculator now lets you activate active modules installed in the selected build. This matters for builds that include active modules such as Stampede.

### 5. Operator / crew context

Sidebar now has:

```text
Operator / pilot
Crew size
Run tag
```

These fields are stored in every event. They are useful for duo collection and later analytics by pilot or run group.

### 6. Resource/yield fields

Each event can now store:

```text
primary_resource
resource_percent
raw_scu_estimate
refined_scu_estimate
estimated_value_auec
mining_time_seconds
resource_comment
```

These values are not used by the current baseline ML model yet. They are collected now so future profit/yield models have data.

## MLOps meaning

This block improves data collection quality before the real dataset exists:

```text
real mining run
  → operator/context
  → rock parameters
  → formula verdict
  → optional ML snapshot
  → actual outcome
  → resource/yield result
  → analytics-ready event
```

The key MLOps improvement is avoiding a common mistake: collecting 100+ rows and only later realizing that resource/profit fields were missing.
