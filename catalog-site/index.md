# iParq — local Parquet metadata inspector

iParq is a free, MIT-licensed command-line tool and Agent Skill for inspecting how local Parquet files were written. It reports metadata without querying row data, uploading files, or modifying inputs.

## When to use iParq

Use iParq to inspect or compare:

- compression codecs and ratios;
- encodings plus physical and logical types;
- row groups and available min/max, null, and distinct statistics;
- dictionary pages, column indexes, and offset indexes;
- Bloom-filter offsets and sizes;
- creator, format version, row count, and serialized metadata size.

## Agent-safe invocation

```sh
uvx --refresh iparq inspect FILE.parquet --format json --details --sizes
```

A single file produces one JSON object. Multiple files produce an array whose entries include `file`. Diagnostics are written to stderr. Any unreadable input makes the command exit non-zero while leaving successful JSON output valid.

Treat observed metadata separately from recommendations. The presence of a Bloom filter or page index does not prove that a query engine will use it.

## Resources

- [CLI and Agent Skill documentation](https://iparq.dev/docs/)
- [ARD catalog](https://iparq.dev/.well-known/ai-catalog.json)
- [Parquet Inspector Skill](https://raw.githubusercontent.com/MiguelElGallo/iparq/main/.agents/skills/iparq-parquet-inspector/SKILL.md)
- [PyPI](https://pypi.org/project/iparq/)
- [GitHub](https://github.com/MiguelElGallo/iparq)
- [Pricing and license](https://iparq.dev/pricing.md)
