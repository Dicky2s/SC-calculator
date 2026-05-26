# Current snapshot

This archive is the current continuation base for the SC Mining Assistant workstream.

Included latest fixes:
- scan-style Resistance/Instability percent inputs are normalized to 0..1 before formula use;
- default scan distance is 15m;
- RockInput validates resistance/instability as normalized fractions;
- Streamlit dataframe rendering is Arrow-safe for mixed value tables;
- scan composition total SCU is stored as `total_scu_estimate`;
- multi-resource and refinery/calibration fields are preserved.
