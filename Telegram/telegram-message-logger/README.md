# Telegram Incoming Message Logger for macOS

This folder is self-contained and can be sent to another Mac with AirDrop. It
signs into that person's own Telegram account, listens only for newly arriving
incoming messages, and writes them to time-windowed `.log` files.

It does not download chat history, capture messages that arrived before startup,
or record outgoing messages.

## Fast setup on the other Mac

1. AirDrop the entire `telegram-message-logger` folder.
2. Get an API ID and API hash from <https://my.telegram.org>.
3. Double-click `start.command`.
4. On first launch, enter the API credentials, the desired number of seconds per
   log file, the Telegram phone number, the Telegram login code, and the
   two-factor password if the account uses one.
5. Keep the Terminal window open.

macOS may block the first launch because the file came from AirDrop. If so,
Control-click `start.command`, choose **Open**, and confirm.

Python 3 must be installed. The first launch creates `.venv` inside this folder
and installs Telethon; later launches reuse it.

## Output format

Logs are created in the `logs` subfolder. A five-minute window looks like:

```text
messages_20260729T180000.000000Z--20260729T180500.000000Z.log
```

Each incoming message is recorded as:

```text
/timestamp 2026-07-29T18:01:02.123Z
/captured_at 2026-07-29T18:01:02.204Z
/who {"id":123456789,"name":"Alice Example","username":"alice"}
/platform Telegram
/message "Exact message text\nincluding line breaks"
```

`/timestamp` is Telegram's message timestamp. `/captured_at` is when this
program received it. JSON quoting on `/who` and `/message` preserves Unicode,
quotes, line breaks, and other special characters precisely. Media without text
is represented by a label such as `[photo]`, `[video]`, or `[document: name]`.
Files themselves are not downloaded.

## Rotation and the Python hook

Set `rotation_seconds` in `config.json`. At every boundary, the program:

1. Closes and flushes the expired `.log`.
2. Immediately opens the next `.log`.
3. Passes the completed path to `process_log_file(log_path)` in
   `postprocess.py` on a background thread.

This ordering means future processing cannot pause message capture. Replace only
the body of `process_log_file` when the final Python function is available.
If that function raises an exception, capture continues and the error appears
in Terminal.

Stopping the logger with Control-C closes the current partial window and sends
that file to the hook as well.

## Configuration

First launch creates `config.json`:

```json
{
  "api_id": 12345678,
  "api_hash": "your_api_hash",
  "rotation_seconds": 300,
  "log_directory": "logs",
  "session_name": "telegram_user"
}
```

Relative paths stay inside this folder. The login is stored locally in
`telegram_user.session`, so a code is normally required only once.

## Important

The `.session` file provides access to the signed-in Telegram account. Do not
AirDrop or share the folder again after signing in. Only monitor an account and
conversations with the account owner's permission.
