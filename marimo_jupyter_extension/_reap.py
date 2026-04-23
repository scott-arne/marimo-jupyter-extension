"""Process-tree-aware wrapper for launching marimo under jupyter-server-proxy.

Marimo spawns its language-server child processes with
``start_new_session=True``, which places them in their own POSIX sessions /
process groups. When the marimo parent is killed (or exits after a failure),
those LSP children are reparented to PID 1 and continue listening on their
TCP ports. Over many Jupyter sessions these orphans accumulate and can
exhaust the 100-port window marimo searches, at which point every new
``marimo edit`` invocation fails with ``RuntimeError: Could not find a free
port``.

This wrapper is spliced in front of the marimo command by
``setup_marimoserver`` so that jupyter-server-proxy sees::

    python -m marimo_jupyter_extension._reap -- <marimo cmd>

On SIGTERM / SIGINT / SIGHUP the wrapper enumerates the descendant tree
*before* signaling (so LSPs still show up as descendants rather than
orphans), SIGTERMs all of them, re-enumerates once to catch anything that
forked during shutdown, gives a short grace period, then SIGKILLs any
stragglers.

On Windows the wrapper is a plain pass-through: marimo does not detach its
children on Windows (``start_new_session=not is_windows()``), so there is no
leak to reap.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

_GRACE_SECONDS = 5.0
_POLL_INTERVAL = 0.1


def _log(msg: str) -> None:
    """Write a single line to stderr (the reaper has no logger)."""
    try:
        sys.stderr.write(f"marimo-jupyter-extension reaper: {msg}\n")
        sys.stderr.flush()
    except OSError:
        pass


def _ps_parent_map() -> dict[int, int]:
    """Return a {pid: ppid} map for every process visible to ``ps``.

    :returns: Mapping from PID to parent PID. Empty if ``ps`` cannot be
        executed for any reason; a warning is written to stderr so that a
        deployment with a broken ``ps`` is visible instead of silently
        falling back to direct-child-only reaping.
    """
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid="],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log(
            f"'ps' invocation failed ({exc!r}); descendant reap degraded "
            "to direct-child-only. Orphan LSPs may leak."
        )
        return {}

    if result.returncode != 0 or not result.stdout:
        _log(
            f"'ps -axo pid=,ppid=' returned rc={result.returncode}; "
            "descendant reap degraded to direct-child-only. "
            "Orphan LSPs may leak."
        )
        return {}

    mapping: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            mapping[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return mapping


def _descendants(root_pid: int) -> set[int]:
    """Return every transitive descendant of ``root_pid`` (excluding itself).

    :param root_pid: PID whose descendant tree is being enumerated.
    :returns: Set of descendant PIDs.
    """
    parent_of = _ps_parent_map()
    children: dict[int, list[int]] = {}
    for pid, ppid in parent_of.items():
        children.setdefault(ppid, []).append(pid)

    result: set[int] = set()
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in result:
            continue
        result.add(pid)
        stack.extend(children.get(pid, []))
    return result


def _is_alive(pid: int) -> bool:
    """Return True if signal 0 can be delivered to ``pid``."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _send(pid: int, sig: int) -> None:
    """Send ``sig`` to ``pid``, swallowing lookup / permission errors."""
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _terminate_tree(root_pid: int, child_pid: int) -> None:
    """Gracefully terminate ``child_pid`` and every descendant of ``root_pid``.

    Snapshots the descendant set, SIGTERMs it, then re-enumerates once to
    catch any process that forked between the original snapshot and the
    first SIGTERM pass (e.g. a late LSP that marimo was still spawning
    when shutdown began). Any processes that survive the grace window
    are SIGKILLed.

    :param root_pid: PID of this wrapper; used as the descendant tree root.
    :param child_pid: PID of the direct marimo child, explicitly included in
        case ``ps`` output is stale.
    """
    targets = _descendants(root_pid) | {child_pid}
    for pid in targets:
        _send(pid, signal.SIGTERM)

    # Second pass: re-enumerate and SIGTERM anything we missed. Processes
    # forked during shutdown (or reparented to PID 1 as their immediate
    # parent died) would otherwise escape the first pass.
    late_targets = _descendants(root_pid) - targets
    if late_targets:
        for pid in late_targets:
            _send(pid, signal.SIGTERM)
        targets |= late_targets

    deadline = time.monotonic() + _GRACE_SECONDS
    while time.monotonic() < deadline:
        if not any(_is_alive(pid) for pid in targets):
            return
        time.sleep(_POLL_INTERVAL)

    for pid in targets:
        if _is_alive(pid):
            _send(pid, signal.SIGKILL)


def _run_windows(argv: list[str]) -> int:
    """Pass-through runner for Windows (no descendant reaping needed)."""
    proc = subprocess.Popen(argv)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 130


def _run_posix(argv: list[str]) -> int:
    """Run ``argv`` and propagate shutdown signals to the descendant tree."""
    proc = subprocess.Popen(argv)

    def handler(_signum: int, _frame: object) -> None:
        _terminate_tree(os.getpid(), proc.pid)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            # Some environments restrict signal installation.
            pass

    try:
        return proc.wait()
    except KeyboardInterrupt:
        _terminate_tree(os.getpid(), proc.pid)
        return 130


def main(argv: list[str]) -> int:
    """Entry point. Accepts ``[--, <cmd>, <args>...]`` or ``[<cmd>, ...]``.

    :param argv: Argument vector *excluding* ``sys.argv[0]``.
    :returns: Exit status of the wrapped command (or 2 on usage error).
    """
    if not argv:
        sys.stderr.write(
            "usage: python -m marimo_jupyter_extension._reap -- "
            "<command> [args ...]\n"
        )
        return 2
    if argv[0] == "--":
        argv = argv[1:]
    if not argv:
        sys.stderr.write(
            "usage: python -m marimo_jupyter_extension._reap -- "
            "<command> [args ...]\n"
        )
        return 2

    if os.name == "nt":
        return _run_windows(argv)
    return _run_posix(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
