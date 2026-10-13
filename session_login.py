"""Local password authentication and participant session storage."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
from uuid import uuid4


@dataclass
class Session:
    session_id: str
    participant_name: str
    started_at: str
    ended_at: str | None = None
    pose_name: str | None = None


class AuthStore:
    """Small local authentication store for the CLI and future GUI."""

    def __init__(self, database_path=None):
        self.database_path = Path(
            database_path or Path(__file__).resolve().parent / "users.db"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _hash_password(password, salt=None):
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1,
        )
        return "$".join((salt.hex(), digest.hex()))

    @staticmethod
    def _check_password(password, stored_hash):
        salt_hex, digest_hex = stored_hash.split("$", 1)
        candidate = AuthStore._hash_password(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(candidate, f"{salt_hex}${digest_hex}")

    def register_user(self, username, password):
        username = username.strip()
        if not username or not password:
            raise ValueError("Username and password are required.")
        password_hash = self._hash_password(password)
        try:
            with sqlite3.connect(self.database_path) as connection:
                connection.execute(
                    "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, datetime.now(timezone.utc).isoformat()),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("That username already exists.") from error
        return username

    def login_user(self, username, password):
        username = username.strip()
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row is None or not self._check_password(password, row[0]):
            raise ValueError("Invalid username or password.")
        return username


def authenticate(database_path=None):
    """Register or authenticate a participant for a new yoga session."""
    store = AuthStore(database_path)
    while True:
        print("\n=== Yoga Session Account ===")
        print("1) Create a new account")
        print("2) Log in to an existing account")
        choice = input("Choose 1 or 2: ").strip()
        if choice not in {"1", "2"}:
            print("Please enter 1 or 2.")
            continue

        print("\nUsername")
        username = input("> ").strip()
        print("Password")
        password = getpass.getpass("> ")
        try:
            if choice == "1":
                store.register_user(username, password)
                print("Account created. You are now logged in.")
                return username
            if choice == "2":
                store.login_user(username, password)
                print("Login successful.")
                return username
        except ValueError as error:
            print(error)


def login(sessions_dir=None):
    """Authenticate a participant and create a local session record."""
    participant_name = authenticate()

    session = Session(
        session_id=uuid4().hex[:8],
        participant_name=participant_name,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    save_session(session, sessions_dir)
    print(f"Session started for {session.participant_name} (ID: {session.session_id})")
    return session


def save_session(session, sessions_dir=None):
    directory = Path(sessions_dir or Path(__file__).resolve().parent / "sessions")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session.session_id}.json"
    path.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8")
    return path


def finish(session, pose_name, sessions_dir=None):
    session.pose_name = pose_name
    session.ended_at = datetime.now(timezone.utc).isoformat()
    return save_session(session, sessions_dir)