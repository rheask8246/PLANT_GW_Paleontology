# Intrinsic Data Validation Summary

Validation source: `all_events.parquet` (intrinsic full-range events; no detectability filter).

| Check | Status | Notes |
|---|---|---|
| `grid_coverage` | **warn** | Low CE occupancy fraction: 0.177 |
| `channel_health` | **pass** |  |
| `event_validity` | **pass** | Optional column 'weight' missing; skipped weight-bound checks. |
| `split_hygiene` | **pass** |  |
| `rare_event_diagnostics` | **pass** |  |
| `distribution_sanity` | **pass** |  |

## Generated Artifacts

- Reports: `test/reports/validation/`
- Plots: `test/plots/validation/`
