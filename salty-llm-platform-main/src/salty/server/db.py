"""SQLite ORM + 初始化"""
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

DB_PATH = Path("./app_data.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT UNIQUE NOT NULL,
                stage TEXT NOT NULL,
                step INTEGER,
                schema_version INTEGER DEFAULT 1,
                is_starred INTEGER DEFAULT 0,
                is_current INTEGER DEFAULT 0,
                params INTEGER,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_ckpt_stage ON checkpoints(stage);
            CREATE INDEX IF NOT EXISTS idx_ckpt_current ON checkpoints(is_current);

            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                stage TEXT NOT NULL,
                config_path TEXT NOT NULL,
                overrides TEXT,
                status TEXT DEFAULT 'pending',
                pid INTEGER,
                start_time TEXT,
                end_time TEXT,
                current_iter INTEGER DEFAULT 0,
                max_iter INTEGER,
                loss REAL,
                log_file TEXT,
                error_msg TEXT,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_job_status ON training_jobs(status);

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                checkpoint_id INTEGER,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                think_content TEXT,
                tokens INTEGER,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()


@dataclass
class CheckpointRow:
    id: int
    name: str
    path: str
    stage: str
    step: Optional[int]
    schema_version: int
    is_starred: bool
    is_current: bool
    params: Optional[int]
    created_at: str


@dataclass
class TrainingJobRow:
    id: int
    run_id: str
    stage: str
    config_path: str
    overrides: Optional[str]
    status: str
    pid: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]
    current_iter: int
    max_iter: Optional[int]
    loss: Optional[float]
    log_file: Optional[str]
    error_msg: Optional[str]
    created_at: str


@dataclass
class ChatSessionRow:
    id: int
    title: Optional[str]
    checkpoint_id: Optional[int]
    created_at: str
    updated_at: str


@dataclass
class ChatMessageRow:
    id: int
    session_id: int
    role: str
    content: str
    think_content: Optional[str]
    tokens: Optional[int]
    created_at: str


class CheckpointDAO:
    @staticmethod
    def insert(row: CheckpointRow) -> int:
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO checkpoints(name,path,stage,step,schema_version,is_starred,is_current,params)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (row.name, row.path, row.stage, row.step, row.schema_version,
                 int(row.is_starred), int(row.is_current), row.params),
            )
            conn.commit()
            return cur.lastrowid

    @staticmethod
    def list_all(stage: Optional[str] = None) -> List[CheckpointRow]:
        with get_conn() as conn:
            if stage:
                rows = conn.execute(
                    "SELECT * FROM checkpoints WHERE stage=? ORDER BY created_at DESC", (stage,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM checkpoints ORDER BY created_at DESC").fetchall()
            return [_row_to_ckpt(r) for r in rows]

    @staticmethod
    def get_current() -> Optional[CheckpointRow]:
        with get_conn() as conn:
            r = conn.execute("SELECT * FROM checkpoints WHERE is_current=1 LIMIT 1").fetchone()
            return _row_to_ckpt(r) if r else None

    @staticmethod
    def set_current(ckpt_id: int):
        with get_conn() as conn:
            conn.execute("UPDATE checkpoints SET is_current=0")
            conn.execute("UPDATE checkpoints SET is_current=1 WHERE id=?", (ckpt_id,))
            conn.commit()

    @staticmethod
    def toggle_star(ckpt_id: int) -> bool:
        with get_conn() as conn:
            r = conn.execute("SELECT is_starred FROM checkpoints WHERE id=?", (ckpt_id,)).fetchone()
            if not r:
                return False
            new_val = 0 if r["is_starred"] else 1
            conn.execute("UPDATE checkpoints SET is_starred=? WHERE id=?", (new_val, ckpt_id))
            conn.commit()
            return bool(new_val)

    @staticmethod
    def delete(ckpt_id: int):
        with get_conn() as conn:
            conn.execute("DELETE FROM checkpoints WHERE id=?", (ckpt_id,))
            conn.commit()


class TrainingJobDAO:
    @staticmethod
    def insert(row: TrainingJobRow) -> int:
        with get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO training_jobs(run_id,stage,config_path,overrides,status,pid,max_iter,log_file)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (row.run_id, row.stage, row.config_path, row.overrides, row.status,
                 row.pid, row.max_iter, row.log_file),
            )
            conn.commit()
            return cur.lastrowid

    @staticmethod
    def list_all() -> List[TrainingJobRow]:
        with get_conn() as conn:
            rows = conn.execute("SELECT * FROM training_jobs ORDER BY created_at DESC").fetchall()
            return [_row_to_job(r) for r in rows]

    @staticmethod
    def get_by_run_id(run_id: str) -> Optional[TrainingJobRow]:
        with get_conn() as conn:
            r = conn.execute("SELECT * FROM training_jobs WHERE run_id=?", (run_id,)).fetchone()
            return _row_to_job(r) if r else None

    @staticmethod
    def update_status(run_id: str, status: str, **kwargs):
        with get_conn() as conn:
            fields = ["status=?"]
            vals = [status]
            for k, v in kwargs.items():
                fields.append(f"{k}=?")
                vals.append(v)
            vals.append(run_id)
            conn.execute(
                f"UPDATE training_jobs SET {','.join(fields)} WHERE run_id=?",
                tuple(vals),
            )
            conn.commit()


class ChatSessionDAO:
    @staticmethod
    def insert(title: Optional[str] = None, checkpoint_id: Optional[int] = None) -> int:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO chat_sessions(title,checkpoint_id) VALUES(?,?)",
                (title, checkpoint_id),
            )
            conn.commit()
            return cur.lastrowid

    @staticmethod
    def list_all() -> List[ChatSessionRow]:
        with get_conn() as conn:
            rows = conn.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC").fetchall()
            return [ChatSessionRow(**{k: r[k] for k in r.keys()}) for r in rows]

    @staticmethod
    def update_title(session_id: int, title: str):
        with get_conn() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title=?, updated_at=datetime('now','localtime') WHERE id=?",
                (title, session_id),
            )
            conn.commit()

    @staticmethod
    def delete(session_id: int):
        with get_conn() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
            conn.commit()


class ChatMessageDAO:
    @staticmethod
    def insert(session_id: int, role: str, content: str, think_content: Optional[str] = None,
               tokens: Optional[int] = None) -> int:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO chat_messages(session_id,role,content,think_content,tokens) VALUES(?,?,?,?,?)",
                (session_id, role, content, think_content, tokens),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at=datetime('now','localtime') WHERE id=?",
                (session_id,),
            )
            conn.commit()
            return cur.lastrowid

    @staticmethod
    def list_by_session(session_id: int) -> List[ChatMessageRow]:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()
            return [ChatMessageRow(**{k: r[k] for k in r.keys()}) for r in rows]


def _row_to_ckpt(r: sqlite3.Row) -> CheckpointRow:
    return CheckpointRow(
        id=r["id"], name=r["name"], path=r["path"], stage=r["stage"],
        step=r["step"], schema_version=r["schema_version"],
        is_starred=bool(r["is_starred"]), is_current=bool(r["is_current"]),
        params=r["params"], created_at=r["created_at"],
    )


def _row_to_job(r: sqlite3.Row) -> TrainingJobRow:
    return TrainingJobRow(
        id=r["id"], run_id=r["run_id"], stage=r["stage"], config_path=r["config_path"],
        overrides=r["overrides"], status=r["status"], pid=r["pid"],
        start_time=r["start_time"], end_time=r["end_time"],
        current_iter=r["current_iter"], max_iter=r["max_iter"],
        loss=r["loss"], log_file=r["log_file"], error_msg=r["error_msg"],
        created_at=r["created_at"],
    )
