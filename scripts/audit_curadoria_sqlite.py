"""Audit text fields in the versioned curation database for obvious secrets/PII.

The audit reports only table and column names, never database values.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path


PATTERNS = {
    "email": re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "credential_word": re.compile(r"password|senha|token|secret", re.IGNORECASE),
    "patient_identifier": re.compile(r"patient|paciente|cpf|nascimento|prontu", re.IGNORECASE),
}


def audit(database: Path) -> list[tuple[str, str, str]]:
    findings = []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute("select name from sqlite_master where type='table'")]
        for table in tables:
            columns = [
                row[1]
                for row in connection.execute(f"pragma table_info({table})")
                if row[2].upper() in {"TEXT", ""}
            ]
            for column in columns:
                for (value,) in connection.execute(f"select {column} from {table} where {column} is not null"):
                    text = str(value)
                    for name, pattern in PATTERNS.items():
                        if pattern.search(text):
                            findings.append((name, table, column))
    finally:
        connection.close()
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("dataset_aumentado/curadoria/respostas/curadoria.sqlite3"),
    )
    args = parser.parse_args()
    database = args.database.resolve()
    if not database.exists():
        print(f"Banco não encontrado: {database}")
        return 1
    findings = audit(database)
    if findings:
        for kind, table, column in findings:
            print(f"{kind}: tabela={table} coluna={column}")
        return 1
    print(f"Auditoria concluída sem padrões suspeitos: {database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
