"""Tests for the ``_reap`` wrapper that reaps marimo's descendant processes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REAP = "marimo_jupyter_extension._reap"


def _pid_alive(pid: int) -> bool:
    """Return True while ``pid`` is a live, non-zombie process."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Treat zombies as dead for test purposes.
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("State:"):
                    return "Z" not in line.split()[1]
    except OSError:
        pass
    return True


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class TestMainArgumentParsing:
    """Argument parsing in ``_reap.main``."""

    def test_returns_error_when_no_args(self):
        from marimo_jupyter_extension._reap import main

        assert main([]) == 2

    def test_returns_error_when_only_separator(self):
        from marimo_jupyter_extension._reap import main

        assert main(["--"]) == 2


class TestPassThrough:
    """End-to-end behavior: exit status and stdout propagate."""

    def test_exit_status_is_propagated(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                REAP,
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ],
            capture_output=True,
        )
        assert result.returncode == 7

    def test_stdout_is_propagated(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                REAP,
                "--",
                sys.executable,
                "-c",
                "print('hello reaper')",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hello reaper" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only descendant reaping")
class TestDescendantReap:
    """Descendants of the wrapped process are killed on SIGTERM."""

    def test_detached_grandchild_is_reaped(self, tmp_path: Path):
        """A grandchild spawned with ``start_new_session=True`` (mimicking
        marimo's LSP spawn) must still be killed when the reaper receives
        SIGTERM."""
        marker = tmp_path / "grandchild.pid"
        # The child writes its detached grandchild's PID to a file, then
        # blocks. When the reaper terminates, both child and grandchild
        # should be gone.
        script = textwrap.dedent(
            f"""
            import os, subprocess, sys, time
            gc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(300)"],
                start_new_session=True,
            )
            open({str(marker)!r}, "w").write(str(gc.pid))
            time.sleep(300)
            """
        )

        proc = subprocess.Popen(
            [sys.executable, "-m", REAP, "--", sys.executable, "-c", script]
        )
        try:
            assert _wait_until(marker.exists), "grandchild never started"
            gc_pid = int(marker.read_text().strip())
            assert _pid_alive(gc_pid)

            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)

            # After the wrapper exits the grandchild must also be gone.
            assert _wait_until(lambda: not _pid_alive(gc_pid), timeout=10), (
                f"grandchild pid={gc_pid} still alive after reaper exit"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            # Belt-and-braces cleanup of the grandchild if the test failed.
            if marker.exists():
                try:
                    os.kill(int(marker.read_text().strip()), signal.SIGKILL)
                except (OSError, ValueError):
                    pass

    def test_late_spawned_grandchild_is_reaped(self, tmp_path: Path):
        """A grandchild spawned *after* SIGTERM arrives (during shutdown)
        must still be killed by the second-pass re-enumeration."""
        marker = tmp_path / "late_grandchild.pid"
        script = textwrap.dedent(
            f"""
            import os, signal, subprocess, sys, time

            def spawn_on_term(signum, frame):
                gc = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(300)"],
                    start_new_session=True,
                )
                open({str(marker)!r}, "w").write(str(gc.pid))
                # Give the reaper's second pass time to see us.
                time.sleep(2.0)
                sys.exit(0)

            signal.signal(signal.SIGTERM, spawn_on_term)
            time.sleep(300)
            """
        )

        proc = subprocess.Popen(
            [sys.executable, "-m", REAP, "--", sys.executable, "-c", script]
        )
        try:
            time.sleep(0.5)
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)

            assert _wait_until(marker.exists, timeout=5), (
                "late grandchild never started"
            )
            gc_pid = int(marker.read_text().strip())

            assert _wait_until(lambda: not _pid_alive(gc_pid), timeout=10), (
                f"late grandchild pid={gc_pid} still alive after reaper exit"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if marker.exists():
                try:
                    os.kill(int(marker.read_text().strip()), signal.SIGKILL)
                except (OSError, ValueError):
                    pass

    def test_clean_exit_does_not_emit_error(self):
        """If the wrapped command exits on its own, the wrapper returns 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                REAP,
                "--",
                sys.executable,
                "-c",
                "pass",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert result.stderr == ""


class TestSetupWiring:
    """``setup_marimoserver`` must splice the reaper in front of marimo."""

    def test_command_starts_with_reaper(self, clean_env, mock_marimo_in_path):
        from marimo_jupyter_extension import setup_marimoserver

        cmd = setup_marimoserver()["command"]

        assert cmd[0] == sys.executable
        assert cmd[1] == "-m"
        assert cmd[2] == "marimo_jupyter_extension._reap"
        assert cmd[3] == "--"

    def test_reaper_is_before_marimo_executable(
        self, clean_env, mock_marimo_in_path
    ):
        """The resolved marimo path must sit after the reaper preamble."""
        from marimo_jupyter_extension import setup_marimoserver

        cmd = setup_marimoserver()["command"]
        marimo_index = cmd.index(mock_marimo_in_path)
        assert marimo_index == 4
        assert "edit" in cmd[marimo_index:]
