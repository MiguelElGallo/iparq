"""Inspect local Vortex files through their schema, footer, and end marker."""

import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel

VORTEX_MAGIC = b"VTXF"
END_MARKER = struct.Struct("<HH4s")


class VortexMetadata(BaseModel):
    """File-level information from the Vortex footer and end marker."""

    file_format: Literal["vortex"] = "vortex"
    format_version: int
    num_rows: int
    num_columns: int
    root_dtype: str
    root_nullable: bool
    file_size_bytes: int
    postscript_size_bytes: int


class VortexColumn(BaseModel):
    """A top-level Vortex field, or the unnamed non-struct root."""

    column_name: str | None
    dtype: str
    nullable: bool


class VortexSplit(BaseModel):
    """A half-open range of rows reported by the Vortex reader."""

    start_row: int
    stop_row: int


class VortexInspection(BaseModel):
    """File metadata, logical fields, and available scan split ranges."""

    metadata: VortexMetadata
    columns: list[VortexColumn]
    splits: list[VortexSplit]


def is_vortex_file(filename: str) -> bool:
    """Identify local files by signature, using the extension for damaged inputs."""
    path = Path(filename)
    if path.is_file():
        with path.open("rb", buffering=0) as stream:
            magic = stream.read(4)
        if magic in (VORTEX_MAGIC, b"PAR1", b"PARE"):
            return magic == VORTEX_MAGIC
    return path.suffix.lower() == ".vortex"


def _read_end_marker(stream: BinaryIO) -> tuple[int, int, int]:
    """Read the version and postscript length from the eight-byte trailer."""
    size = stream.seek(0, os.SEEK_END)
    if size < len(VORTEX_MAGIC) + END_MARKER.size:
        raise ValueError("File is too short for a Vortex end marker")
    stream.seek(-END_MARKER.size, os.SEEK_END)
    version, postscript_size, magic = END_MARKER.unpack(stream.read(END_MARKER.size))
    if magic != VORTEX_MAGIC or postscript_size > size - END_MARKER.size - 4:
        raise ValueError("Invalid Vortex end marker or postscript length")
    stream.seek(0)
    if stream.read(4) != VORTEX_MAGIC:
        raise ValueError("Invalid Vortex file signature")
    return version, postscript_size, size


if sys.version_info >= (3, 11):

    def read_vortex(
        filename: str, *, include_columns: bool, include_splits: bool
    ) -> VortexInspection:
        """Read the local file schema and requested split metadata without scanning rows."""
        try:
            import vortex as vx
        except ModuleNotFoundError as error:
            if error.name != "vortex":
                raise
            raise RuntimeError(
                "Install Vortex support with: pip install 'iparq[vortex]'"
            ) from error

        with open(filename, "rb") as stream:
            version, postscript_size, size = _read_end_marker(stream)
            file = vx.open_readable(stream)
            dtype = file.dtype
            fields: list[tuple[str | None, vx.DType]]
            if isinstance(dtype, vx.StructDType):
                # The native names() method returns display-escaped field names.
                names = (
                    dtype.to_arrow_schema().names if include_columns else dtype.names()
                )
                num_columns = len(names)
                root_dtype = "struct"
                fields = list(zip(names, dtype.fields())) if include_columns else []
            else:
                num_columns = 1
                root_dtype = str(dtype)
                fields = [(None, dtype)] if include_columns else []
            metadata = VortexMetadata(
                format_version=version,
                num_rows=file.footer.row_count,
                num_columns=num_columns,
                root_dtype=root_dtype,
                root_nullable=dtype.is_nullable(),
                file_size_bytes=size,
                postscript_size_bytes=postscript_size,
            )
            columns = [
                VortexColumn(
                    column_name=name, dtype=str(field), nullable=field.is_nullable()
                )
                for name, field in fields
            ]
            splits = [
                VortexSplit(start_row=start, stop_row=stop)
                for start, stop in (file.splits() if include_splits else ())
            ]
        return VortexInspection(metadata=metadata, columns=columns, splits=splits)

else:

    def read_vortex(
        filename: str, *, include_columns: bool, include_splits: bool
    ) -> VortexInspection:
        """Explain the minimum Python version required by the native Vortex reader."""
        raise RuntimeError("Vortex inspection requires Python 3.11 or newer")
