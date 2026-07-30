#!/usr/bin/env python3
"""Capture new incoming Telegram messages from a signed-in user account."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

from telethon import TelegramClient, events, utils
from telethon.tl.custom.message import Message

from postprocess import process_log_file


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOGGER = logging.getLogger("telegram_logger")


@dataclass(frozen=True)
class Config:
    api_id: int
    api_hash: str
    rotation_seconds: int
    log_directory: Path
    session_path: Path


@dataclass(frozen=True)
class LogEntry:
    captured_at: datetime
    text: str


def prompt_for_config() -> dict[str, object]:
    print("First-time Telegram logger setup")
    print("Create API credentials at https://my.telegram.org, then enter them here.")
    api_id = input("API ID: ").strip()
    api_hash = input("API hash: ").strip()
    rotation = input("Seconds per log file [300]: ").strip() or "300"

    raw_config: dict[str, object] = {
        "api_id": int(api_id),
        "api_hash": api_hash,
        "rotation_seconds": int(rotation),
        "log_directory": "logs",
        "session_name": "telegram_user",
    }
    CONFIG_PATH.write_text(
        json.dumps(raw_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {CONFIG_PATH}")
    return raw_config


def load_config() -> Config:
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    elif sys.stdin.isatty():
        raw = prompt_for_config()
    else:
        raise RuntimeError(
            f"{CONFIG_PATH} is missing; copy config.example.json to config.json"
        )

    try:
        api_id = int(raw["api_id"])
        api_hash = str(raw["api_hash"]).strip()
        rotation_seconds = int(raw.get("rotation_seconds", 300))
        log_directory = str(raw.get("log_directory", "logs")).strip()
        session_name = str(raw.get("session_name", "telegram_user")).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration in {CONFIG_PATH}: {exc}") from exc

    if api_id <= 0 or not api_hash:
        raise ValueError("api_id and api_hash must be set in config.json")
    if rotation_seconds <= 0:
        raise ValueError("rotation_seconds must be greater than zero")
    if not log_directory or not session_name:
        raise ValueError("log_directory and session_name cannot be empty")

    log_path = Path(log_directory).expanduser()
    if not log_path.is_absolute():
        log_path = BASE_DIR / log_path

    session_path = Path(session_name).expanduser()
    if not session_path.is_absolute():
        session_path = BASE_DIR / session_path

    return Config(
        api_id=api_id,
        api_hash=api_hash,
        rotation_seconds=rotation_seconds,
        log_directory=log_path,
        session_path=session_path,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def file_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def message_text(message: Message) -> str:
    if message.raw_text:
        return message.raw_text
    if message.photo:
        return "[photo]"
    if message.video:
        return "[video]"
    if message.voice:
        return "[voice]"
    if message.video_note:
        return "[video_note]"
    if message.audio:
        return "[audio]"
    if message.gif:
        return "[gif]"
    if message.sticker:
        return "[sticker]"
    if message.document:
        name = getattr(message.file, "name", None)
        return f"[document: {name}]" if name else "[document]"
    if message.contact:
        return "[contact]"
    if message.geo:
        return "[location]"
    if message.poll:
        return "[poll]"
    return "[service_or_unsupported_message]"


async def format_entry(event: events.NewMessage.Event) -> LogEntry:
    captured_at = utc_now()
    message = event.message
    sender = event.sender
    if sender is None:
        sender = await event.get_sender()

    sender_id = utils.get_peer_id(sender) if sender is not None else message.sender_id
    username = getattr(sender, "username", None)
    display_name = utils.get_display_name(sender) if sender is not None else None
    who = {
        "id": sender_id,
        "name": display_name or None,
        "username": username or None,
    }

    lines = (
        f"/timestamp {timestamp(message.date)}\n"
        f"/captured_at {timestamp(captured_at)}\n"
        f"/who {json.dumps(who, ensure_ascii=False, separators=(',', ':'))}\n"
        "/platform Telegram\n"
        f"/message {json.dumps(message_text(message), ensure_ascii=False)}\n\n"
    )
    return LogEntry(captured_at=captured_at, text=lines)


class TimedLogWriter:
    """Single-writer queue that rotates files without blocking event handlers."""

    def __init__(self, directory: Path, rotation_seconds: int) -> None:
        self.directory = directory
        self.window = timedelta(seconds=rotation_seconds)
        self.queue: asyncio.Queue[LogEntry | None] = asyncio.Queue()
        self._file: TextIO | None = None
        self._path: Path | None = None
        self._window_start: datetime | None = None
        self._window_end: datetime | None = None
        self._callback_tasks: set[asyncio.Task[None]] = set()

    async def put(self, entry: LogEntry) -> None:
        self.queue.put_nowait(entry)

    def _open_window(self, start: datetime) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._window_start = start
        self._window_end = start + self.window
        filename = (
            f"messages_{file_timestamp(self._window_start)}"
            f"--{file_timestamp(self._window_end)}.log"
        )
        self._path = self.directory / filename
        self._file = self._path.open("a", encoding="utf-8", buffering=64 * 1024)
        LOGGER.info("Writing %s", self._path)

    def _dispatch_callback(self, path: Path) -> None:
        async def run_callback() -> None:
            try:
                await asyncio.to_thread(process_log_file, path)
            except Exception:
                LOGGER.exception("Post-processing failed for %s", path)

        task = asyncio.create_task(run_callback())
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)

    def _close_window(self, send_to_callback: bool = True) -> Path | None:
        if self._file is None or self._path is None:
            return None
        path = self._path
        self._file.flush()
        self._file.close()
        self._file = None
        self._path = None
        if send_to_callback:
            self._dispatch_callback(path)
        return path

    def _rotate(self, next_start: datetime) -> None:
        expired_path = self._close_window(send_to_callback=False)
        self._open_window(next_start)
        if expired_path is not None:
            self._dispatch_callback(expired_path)

    async def run(self) -> None:
        self._open_window(utc_now())
        stopping = False

        while not stopping:
            assert self._window_end is not None
            timeout = max(0.0, (self._window_end - utc_now()).total_seconds())
            try:
                entry = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                self._rotate(self._window_end)
                continue

            if entry is None:
                stopping = True
                continue

            assert self._window_end is not None
            while entry.captured_at >= self._window_end:
                self._rotate(self._window_end)

            assert self._file is not None
            self._file.write(entry.text)
            self._file.flush()

        self._close_window()
        if self._callback_tasks:
            await asyncio.gather(*self._callback_tasks, return_exceptions=True)

    async def stop(self) -> None:
        self.queue.put_nowait(None)


class PerMessageLogWriter:
    """Writes each incoming message to its own .log file immediately."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._callback_tasks: set[asyncio.Task[None]] = set()

    def _dispatch_callback(self, path: Path) -> None:
        async def run_callback() -> None:
            try:
                await asyncio.to_thread(process_log_file, path)
            except Exception:
                LOGGER.exception("Post-processing failed for %s", path)

        task = asyncio.create_task(run_callback())
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)

    async def put(self, entry: LogEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = file_timestamp(entry.captured_at)
        path = self.directory / f"messages_{stamp}.log"
        path.write_text(entry.text, encoding="utf-8")
        LOGGER.info("Wrote %s", path)
        self._dispatch_callback(path)

    async def stop(self) -> None:
        if self._callback_tasks:
            await asyncio.gather(*self._callback_tasks, return_exceptions=True)


async def run() -> None:
    config = load_config()
    writer = PerMessageLogWriter(config.log_directory)

    client = TelegramClient(
        str(config.session_path),
        config.api_id,
        config.api_hash,
        sequential_updates=False,
    )

    @client.on(events.NewMessage(incoming=True))
    async def on_new_message(event: events.NewMessage.Event) -> None:
        try:
            await writer.put(await format_entry(event))
        except Exception:
            LOGGER.exception("Could not capture incoming message")

    try:
        await client.start()
        me = await client.get_me()
        LOGGER.info(
            "Signed in as %s (id=%s); only new incoming messages will be captured",
            utils.get_display_name(me),
            me.id,
        )
        await client.run_until_disconnected()
    finally:
        await writer.stop()
        await client.disconnect()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
