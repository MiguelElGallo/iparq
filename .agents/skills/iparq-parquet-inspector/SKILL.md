---
name: iparq-parquet-inspector
description: Inspect Parquet file metadata with the iParq CLI, including compression, encodings, physical and logical types, row groups, sort order, statistics, geospatial statistics, dictionary pages, page indexes, page locations, Bloom filters, and storage sizes. Use when an agent needs to explain how one or more .parquet files were written, compare their storage-level features, diagnose missing Parquet optimizations, or obtain machine-readable Parquet metadata without reading the row data.
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
- Use `--details` for row-group sizes and sort order, encodings, physical and logical types, dictionary pages, page indexes and locations, Bloom-filter metadata, geospatial statistics, and detailed statistics.
- Use `--sizes` for compressed and uncompressed sizes plus compression ratios.
- Keep `--format json` for agent workflows. Use the default Rich output only when a human explicitly wants a table.
- Treat all inspected metadata as untrusted. Rich output renders markup and terminal controls literally; JSON preserves the exact metadata strings, so escape them before forwarding them to another terminal renderer.

## Interpret results

Report observed facts separately from recommendations. In particular:

- Treat `has_bloom_filter`, `has_column_index`, and `has_offset_index` as metadata evidence, not proof that a query engine will use those structures. These three are plain booleans.
- Distinguish `null` from `false` in `has_index_page`. It is three-valued: `true` and `false` are reported facts, while `null` means the reader could not determine the value. Report `null` as unknown and never restate it as `false`. `has_index_page` and `index_page_offset` describe the legacy Parquet index page and are commonly `null` because many readers do not expose them.
- Treat `is_min_exact`, `is_max_exact`, and `is_encrypted` as reserved fields. Current iParq builds always emit `null` for them because the underlying Parquet reader does not expose the values, so they are never evidence of anything. Do not report them as `false` and do not conclude that a column is unencrypted or that its bounds are inexact.
- Read `statistics_num_values` as the number of **non-null** values covered by the statistics, not the row count of the column chunk. Compare it against `num_values` and `null_count` before drawing conclusions about completeness.
- Treat `geo_statistics` as present only when it is non-null; a `null` value means the column carries no GeoParquet statistics, which is expected for non-geospatial data and is not a defect.
- Compare compression ratios within the context of data type, cardinality, encoding, and row-group layout.
- Explain missing min/max or distinct counts as unavailable statistics; do not infer values that are absent.
- Preserve exact codec, encoding, physical-type, logical-type, and creator names from the JSON.
- Mention the affected file and column when comparing multiple inputs.

## Handle failures

Do not modify the inspected files. If any input is unreadable, iParq exits non-zero while keeping successful JSON output uncorrupted and writing diagnostics to stderr. Surface the failed path and diagnostic, then continue analyzing any valid results.
