"""SQLite-backed lineage storage utilities."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


class LineageDatabase:
    """Thin wrapper around SQLite for lineage data persistence."""

    def __init__(self, db_path: Path, *, reset: bool = False) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.db_path.exists():
            self.db_path.unlink()

        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER REFERENCES agents(id),
                    root_id INTEGER NOT NULL,
                    generation INTEGER NOT NULL,
                    birth INTEGER NOT NULL,
                    death INTEGER
                );
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS founders (
                    id INTEGER PRIMARY KEY
                );
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agents_root_alive
                ON agents(root_id)
                WHERE death IS NULL;
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agents_parent
                ON agents(parent_id);
                """
            )

    # ------------------------------------------------------------------
    # Inserts / updates
    # ------------------------------------------------------------------
    def register_founder(self, agent_id: int, birth: int) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO agents (id, parent_id, root_id, generation, birth, death)
                VALUES (?, NULL, ?, 0, ?, NULL);
                """,
                (agent_id, agent_id, birth),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO founders (id) VALUES (?);",
                (agent_id,),
            )

    def register_offspring(
        self,
        agent_id: int,
        parent_id: int,
        root_id: int,
        generation: int,
        birth: int,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO agents (id, parent_id, root_id, generation, birth, death)
                VALUES (?, ?, ?, ?, ?, NULL);
                """,
                (agent_id, parent_id, root_id, generation, birth),
            )

    def mark_deaths(self, agent_ids: Sequence[int], death_timestep: int) -> None:
        if not agent_ids:
            return
        with self.conn:
            self.conn.executemany(
                "UPDATE agents SET death = ? WHERE id = ? AND death IS NULL;",
                [(death_timestep, agent_id) for agent_id in agent_ids],
            )

    def update_generations(self, updates: Sequence[Tuple[int, int]]) -> None:
        """Update generation values for existing agents."""
        if not updates:
            return
        with self.conn:
            self.conn.executemany(
                "UPDATE agents SET generation = ? WHERE id = ?;",
                [(generation, agent_id) for agent_id, generation in updates],
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def living_ids(self) -> List[int]:
        cursor = self.conn.execute("SELECT id FROM agents WHERE death IS NULL;")
        return [row[0] for row in cursor.fetchall()]

    def root_counts(self) -> Dict[int, int]:
        cursor = self.conn.execute(
            """
            SELECT root_id, COUNT(*)
            FROM agents
            WHERE death IS NULL
            GROUP BY root_id;
            """
        )
        return {root: count for root, count in cursor.fetchall()}

    def founders(self) -> List[int]:
        cursor = self.conn.execute("SELECT id FROM founders ORDER BY id;")
        return [row[0] for row in cursor.fetchall()]

    def living_generations(self) -> List[int]:
        cursor = self.conn.execute("SELECT generation FROM agents WHERE death IS NULL;")
        return [row[0] for row in cursor.fetchall()]

    def max_agent_id(self) -> int:
        cursor = self.conn.execute("SELECT MAX(id) FROM agents;")
        row = cursor.fetchone()
        if not row or row[0] is None:
            return -1
        return int(row[0])

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def vacuum(self) -> None:
        with self.conn:
            self.conn.execute("VACUUM;")

    def close(self) -> None:
        try:
            with self.conn:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            self.conn.close()


def ensure_database(path: Path, *, reset: bool = False) -> LineageDatabase:
    """Convenience wrapper to create a database instance."""
    return LineageDatabase(path, reset=reset)
