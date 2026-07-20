---
name: iparq-parquet-inspector
description: Inspect Parquet file metadata with the iParq CLI, including compression, encodings, physical and logical types, row groups, statistics, dictionary pages, page indexes, Bloom filters, and storage sizes. Use when an agent needs to explain how one or more .parquet files were written, compare their storage-level features, diagnose missing Parquet optimizations, or obtain machine-readable Parquet metadata without reading the row data.
---

# iParq Parquet Inspector

Inspect Parquet storage metadata without querying the file's row data. Prefer JSON output so results remain machine-readable and diagnostics stay separate on stderr.

## Run an inspection

Use the published package without installing it permanently:

```sh
uvx --refresh iparq inspect FILE.parquet --format json --details --sizes
```

If `iparq` is already installed, run it directly:

```sh
iparq inspect FILE.parquet --format json --details --sizes
```

Pass multiple paths or shell-expanded glob patterns to compare files. iParq emits one JSON object for a single file and an array with a `file` field for multiple files.

## Select the minimum useful detail

- Use `--metadata-only` for creator, row count, row groups, Parquet version, and serialized metadata size.
- Use `--column NAME` to restrict column-level output.
- Use `--details` for encodings, physical and logical types, dictionary pages, page indexes, Bloom-filter metadata, and detailed statistics.
- Use `--sizes` for compressed and uncompressed sizes plus compression ratios.
- Keep `--format json` for agent workflows. Use the default Rich output only when a human explicitly wants a table.

## Interpret results

Report observed facts separately from recommendations. In particular:

- Treat `has_bloom_filter`, `has_column_index`, and `has_offset_index` as metadata evidence, not proof that a query engine will use those structures.
- Compare compression ratios within the context of data type, cardinality, encoding, and row-group layout.
- Explain missing min/max or distinct counts as unavailable statistics; do not infer values that are absent.
- Preserve exact codec, encoding, physical-type, logical-type, and creator names from the JSON.
- Mention the affected file and column when comparing multiple inputs.

## Handle failures

Do not modify the inspected files. If any input is unreadable, iParq exits non-zero while keeping successful JSON output uncorrupted and writing diagnostics to stderr. Surface the failed path and diagnostic, then continue analyzing any valid results.
