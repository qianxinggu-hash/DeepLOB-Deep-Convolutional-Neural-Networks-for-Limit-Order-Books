# DeepLOB reproduction report notes

Audience: technical reader evaluating what was and was not reproduced.

## Chart map

- `matched_7709_macro_f1_chart`: compact horizontal bar; compares Macro F1 for FI-2010, LOBSTER, and majority baseline on the exact same 128,422 labels. A single blue root and direct category labels keep the ranking readable without implying that the local LOBSTER holdout is comparable to the paper's LSE test.
- `matched_7709_class_distribution_chart`: grouped bar; compares the actual down/stationary/up shares with FI-2010 and LOBSTER predictions on the same 128,422 samples. A grouped comparison is used instead of a 100% stack because the main question is model-versus-label deviation within each class.
- `matched_7709_classification_report_table`: exact lookup table; records per-class precision, recall, F1 and support for both checkpoints, plus overall accuracy. Macro/weighted summary rows were omitted at the user's request.
- `deeplob_architecture_figure`: generated architecture schematic derived from paper Figures 3 and 4; the editable Mermaid source is `output/report/deeplob_theory_architecture.md`.
- `deeplob_architecture_table`: exact architecture lookup table; keeps filter sizes, branch structure, and component roles auditable next to the schematic.
- `deeplob_data_processing_table`: method comparison table; separates the paper's FI-2010 and LSE pipelines from the public notebook path used for the local FI checkpoint.
- `fi2010_paper_vs_local_table`: exact FI-2010 comparison; uses the same `k=100` horizon but explicitly flags the different split and the fact that the local run reloads a checkpoint rather than retraining.
- `state_example_figure`: custom two-panel bilateral depth chart built from two real 7709 sampled snapshots. Price-positioned bars, a shared quantity scale, bid/ask colors, and mid-price guides make one 40-feature state easier to read than a generic native chart.

## Omitted chart

No chart directly overlays the local LOBSTER holdout result with the paper LSE result. The protocols differ in exchange, period, sample scale, labeling, normalization, and split, so a shared visual scale would encourage a false ranking. Exact values remain in adjacent tables with an explicit comparability note.

## Validation notes

- Paper Tables I, II, and IV were checked by both `pdftotext -layout` and rendered PDF pages.
- The FI-2010 and LOBSTER checkpoints were evaluated on the same 128,422 7709 target indices and label counts `[43,951, 40,094, 44,377]` for the matched comparison.
- Local result values are read from completed JSON artifacts; no notebook code is shown in the reader-facing report.
- The state example is reproducibly extracted from snapshot indices 27,802 and 27,803 in `data/processed/hk07709_2026-07-09_lob.npz`; the embedded PNG and accompanying L1 example are derived from `output/results/7709_state_example.json`.
