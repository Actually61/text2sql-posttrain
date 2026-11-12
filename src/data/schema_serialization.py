"""Render one Spider schema as readable model input, without reading any files.

This is a description of Spider's metadata, not executable CREATE TABLE DDL.
Column indices in primary_keys/foreign_keys always address the original column
array, including the wildcard placeholder at index 0.
"""
from __future__ import annotations

from typing import Any


REQUIRED_FIELDS = (
    "db_id", "table_names_original", "column_names_original", "column_types",
    "primary_keys", "foreign_keys",
)


def _integer(value: Any) -> bool:
    # bool is an int subclass, but cannot be a schema index.
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_schema(schema: Any) -> None:
    """Raise ValueError for malformed metadata rather than dropping a field."""
    if not isinstance(schema, dict):
        raise ValueError("schema must be a dictionary")
    missing = [name for name in REQUIRED_FIELDS if name not in schema]
    if missing:
        raise ValueError(f"Missing schema fields: {', '.join(missing)}")
    if not _nonempty_string(schema["db_id"]):
        raise ValueError("db_id must be a nonempty string")

    tables = schema["table_names_original"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("table_names_original must be a nonempty list")
    if not all(_nonempty_string(name) for name in tables):
        raise ValueError("Every original table name must be a nonempty string")

    columns = schema["column_names_original"]
    types = schema["column_types"]
    if not isinstance(columns, list) or not columns:
        raise ValueError("column_names_original must be a nonempty list")
    if not isinstance(types, list) or len(types) != len(columns):
        raise ValueError("column_types must match the original column array length")
    if not all(_nonempty_string(label) for label in types):
        raise ValueError("Every column type must be a nonempty metadata label")

    for index, column in enumerate(columns):
        if not isinstance(column, (list, tuple)) or len(column) != 2:
            raise ValueError(f"Column index {index} must contain table_id and name")
        table_id, name = column
        if not _integer(table_id) or not _nonempty_string(name):
            raise ValueError(f"Invalid table_id or original name at column index {index}")
        if index == 0:
            if table_id != -1 or name != "*":
                raise ValueError("Column index 0 must be the [-1, '*'] placeholder")
        elif not 0 <= table_id < len(tables):
            raise ValueError(f"Column index {index} has an invalid table_id")

    def check_column_index(index: Any, field: str) -> None:
        if not _integer(index) or not 1 <= index < len(columns):
            raise ValueError(f"{field} must reference a real original column index, never index 0")

    primary_keys = schema["primary_keys"]
    if not isinstance(primary_keys, list):
        raise ValueError("primary_keys must be a list of original column indices")
    for index in primary_keys:
        check_column_index(index, "primary_keys")

    foreign_keys = schema["foreign_keys"]
    if not isinstance(foreign_keys, list):
        raise ValueError("foreign_keys must be a list of original column index pairs")
    for pair in foreign_keys:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Every foreign_keys entry must have source and target indices")
        check_column_index(pair[0], "foreign_keys source")
        check_column_index(pair[1], "foreign_keys target")


def _quote_identifier(name: str) -> str:
    """Keep original spelling; double embedded double quotes for clarity."""
    return '"' + name.replace('"', '""') + '"'


def serialize_schema(schema: dict[str, Any]) -> str:
    """Return stable structure-only text in original table/column order."""
    validate_schema(schema)
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]
    lines = [
        f"Database: {schema['db_id']}",
        "Column types are Spider metadata labels, not exact SQLite DDL types.",
    ]
    for table_id, table_name in enumerate(tables):
        lines.extend(["", f"Table {_quote_identifier(table_name)}:", "  Columns:"])
        # Do not renumber after omitting index 0: PK/FK still use original indices.
        for column_index, (owner_id, column_name) in enumerate(columns):
            if column_index != 0 and owner_id == table_id:
                lines.append(f"    {_quote_identifier(column_name)} [{schema['column_types'][column_index]}]")

        key_names = [
            _quote_identifier(columns[index][1])
            for index in schema["primary_keys"] if columns[index][0] == table_id
        ]
        lines.append("  Primary key columns: " + (", ".join(key_names) if key_names else "(none recorded)"))
        links = [pair for pair in schema["foreign_keys"] if columns[pair[0]][0] == table_id]
        if links:
            lines.append("  Foreign keys:")
            for source_index, target_index in links:
                source_name = columns[source_index][1]
                target_table_id, target_name = columns[target_index]
                lines.append(
                    f"    {_quote_identifier(source_name)} -> "
                    f"{_quote_identifier(tables[target_table_id])}.{_quote_identifier(target_name)}"
                )
        else:
            lines.append("  Foreign keys: (none recorded)")
    return "\n".join(lines)
