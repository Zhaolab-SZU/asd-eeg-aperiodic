# Submitted Supplementary Tables

These CSV files are a machine-readable export of the tables in the Supplementary Information submitted to Nature Mental Health on 28 September 2026. They are numerical locks for the public code release, not newly computed outputs.

The source DOCX is not distributed in this repository. Authors can regenerate the snapshot with:

```bash
python scripts/release/export_submitted_tables.py /path/to/Supplementary_Information_NMH_final.docx
```

The Word document contains 20 table objects because Supplementary Table S1 and Supplementary Table S17 each contain two parts. The exporter checks that all 20 objects are present. Values are copied as displayed; line breaks and repeated whitespace are normalized only for CSV readability.

When a recomputation differs from these tables, do not overwrite the submitted lock silently. Record the discrepancy, the input hashes and the reason, then resolve it against the analysis workspace.
