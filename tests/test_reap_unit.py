"""In-process unit tests for ``_reap`` helpers.

The end-to-end tests in ``test_reap.py`` exercise ``_reap`` as a subprocess,
which means their code paths do not register with the parent coverage
collector. These tests import and call the helpers directly so that the
reaper's internals are visible to coverage.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from marimo_jupyter_extension import _reap


class TestLog:
    """Stderr logger does not raise when stderr is unavailable."""

    def test_writes_prefixed_message(self, capsys):
        _reap._log("hello")
        captured = capsys.readouterr()
        assert "marimo-jupyter-extension reaper: hello" in captured.err

    def test_oserror_on_stderr_is_swallowed(self):
        with patch.object(_reap.sys, "stderr") as stderr:
            stderr.write.side_effect = OSError("closed")
            _reap._log("unreachable")


class TestPsParentMap:
    """``_ps_parent_map`` parses ``ps`` output and handles failures."""

    def test_parses_real_ps_output(self):
        mapping = _reap._ps_parent_map()
        assert mapping, "ps should return at least one process on a live host"
        assert os.getpid() in mapping

    def test_returns_empty_when_ps_missing(self):
        with patch.object(_reap.subprocess, "run", side_effect=OSError):
            assert _reap._ps_parent_map() == {}

    def test_returns_empty_when_ps_raises_subprocess_error(self):
        with patch.object(
            _reap.subprocess,
            "run",
            side_effect=subprocess.SubprocessError("boom"),
        ):
            assert _reap._ps_parent_map() == {}

    def test_returns_empty_when_ps_returncode_nonzero(self):
        fake = MagicMock(returncode=1, stdout="")
        with patch.object(_reap.subprocess, "run", return_value=fake):
            assert _reap._ps_parent_map() == {}

    def test_skips_malformed_lines(self):
        fake = MagicMock(
            returncode=0,
            stdout="1 0\n2 1\nnot a row\nabc def\n3 2\n",
        )
        with patch.object(_reap.subprocess, "run", return_value=fake):
            assert _reap._ps_parent_map() == {1: 0, 2: 1, 3: 2}


class TestDescendants:
    """``_descendants`` walks the parent map transitively."""

    def test_returns_transitive_descendants(self):
        parent_map = {10: 1, 20: 10, 30: 20, 40: 1, 50: 40}
        with patch.object(_reap, "_ps_parent_map", return_value=parent_map):
            assert _reap._descendants(10) == {20, 30}

    def test_returns_empty_for_leaf(self):
        with patch.object(_reap, "_ps_parent_map", return_value={1: 0}):
            assert _reap._descendants(99) == set()


class TestIsAlive:
    """``_is_alive`` uses signal 0 and swallows errors."""

    def test_self_is_alive(self):
        assert _reap._is_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self):
        with patch.object(_reap.os, "kill", side_effect=ProcessLookupError):
            assert _reap._is_alive(999999) is False

    def test_permission_error_means_not_alive(self):
        with patch.object(_reap.os, "kill", side_effect=PermissionError):
            assert _reap._is_alive(1) is False

    def test_other_os_error_means_not_alive(self):
        with patch.object(_reap.os, "kill", side_effect=OSError("unexpected")):
            assert _reap._is_alive(1) is False


class TestSend:
    """``_send`` swallows lookup / permission / OS errors."""

    def test_sends_signal(self):
        with patch.object(_reap.os, "kill") as kill:
            _reap._send(42, signal.SIGTERM)
            kill.assert_called_once_with(42, signal.SIGTERM)

    def test_swallows_process_lookup_error(self):
        with patch.object(_reap.os, "kill", side_effect=ProcessLookupError):
            _reap._send(42, signal.SIGTERM)

    def test_swallows_permission_error(self):
        with patch.object(_reap.os, "kill", side_effect=PermissionError):
            _reap._send(42, signal.SIGTERM)

    def test_swallows_os_error(self):
        with patch.object(_reap.os, "kill", side_effect=OSError("nope")):
            _reap._send(42, signal.SIGTERM)


class TestTerminateTree:
    """``_terminate_tree`` signals descendants, re-enumerates, then SIGKILLs."""

    def test_sigterms_initial_descendants_and_child(self):
        with (
            patch.object(
                _reap, "_descendants", side_effect=[{100, 101}, set()]
            ),
            patch.object(_reap, "_send") as send,
            patch.object(_reap, "_is_alive", return_value=False),
        ):
            _reap._terminate_tree(root_pid=1, child_pid=200)

        sigterm_targets = {
            args[0]
            for args, _ in send.call_args_list
            if args[1] == signal.SIGTERM
        }
        assert sigterm_targets == {100, 101, 200}

    def test_second_pass_catches_late_spawns(self):
        with (
            patch.object(
                _reap, "_descendants", side_effect=[{100}, {100, 101}]
            ),
            patch.object(_reap, "_send") as send,
            patch.object(_reap, "_is_alive", return_value=False),
        ):
            _reap._terminate_tree(root_pid=1, child_pid=200)

        sigterm_targets = [
            args[0]
            for args, _ in send.call_args_list
            if args[1] == signal.SIGTERM
        ]
        assert sorted(sigterm_targets) == [100, 101, 200]

    def test_sigkills_survivors(self, monkeypatch):
        monkeypatch.setattr(_reap, "_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(_reap, "_POLL_INTERVAL", 0.01)
        with (
            patch.object(_reap, "_descendants", side_effect=[{100}, set()]),
            patch.object(_reap, "_send") as send,
            patch.object(_reap, "_is_alive", return_value=True),
        ):
            _reap._terminate_tree(root_pid=1, child_pid=200)

        sigkill_targets = {
            args[0]
            for args, _ in send.call_args_list
            if args[1] == signal.SIGKILL
        }
        assert sigkill_targets == {100, 200}

    def test_returns_early_when_all_targets_dead(self, monkeypatch):
        monkeypatch.setattr(_reap, "_GRACE_SECONDS", 5.0)
        monkeypatch.setattr(_reap, "_POLL_INTERVAL", 0.01)
        with (
            patch.object(_reap, "_descendants", side_effect=[{100}, set()]),
            patch.object(_reap, "_send") as send,
            patch.object(_reap, "_is_alive", return_value=False),
        ):
            _reap._terminate_tree(root_pid=1, child_pid=200)

        sigkill_calls = [
            c for c in send.call_args_list if c.args[1] == signal.SIGKILL
        ]
        assert sigkill_calls == []


class TestRunWindows:
    """``_run_windows`` is a pass-through with KeyboardInterrupt handling."""

    def test_returns_child_exit_code(self):
        proc = MagicMock()
        proc.wait.return_value = 0
        with patch.object(_reap.subprocess, "Popen", return_value=proc):
            assert _reap._run_windows(["echo", "hi"]) == 0

    def test_keyboard_interrupt_terminates_child(self):
        proc = MagicMock()
        proc.wait.side_effect = KeyboardInterrupt
        with patch.object(_reap.subprocess, "Popen", return_value=proc):
            assert _reap._run_windows(["sleep", "9"]) == 130
        proc.terminate.assert_called_once()


class TestRunPosix:
    """``_run_posix`` installs handlers and propagates on KeyboardInterrupt."""

    def test_installs_handlers_and_returns_child_exit_code(self):
        proc = MagicMock()
        proc.wait.return_value = 7
        proc.pid = 12345
        installed: dict[int, object] = {}

        def fake_signal(sig, handler):
            installed[sig] = handler

        with (
            patch.object(_reap.subprocess, "Popen", return_value=proc),
            patch.object(_reap.signal, "signal", side_effect=fake_signal),
        ):
            rc = _reap._run_posix(["echo", "hi"])
        assert rc == 7
        assert signal.SIGTERM in installed
        assert signal.SIGINT in installed
        assert signal.SIGHUP in installed

    def test_signal_installation_errors_are_swallowed(self):
        proc = MagicMock()
        proc.wait.return_value = 0
        with (
            patch.object(_reap.subprocess, "Popen", return_value=proc),
            patch.object(
                _reap.signal, "signal", side_effect=OSError("restricted")
            ),
        ):
            assert _reap._run_posix(["echo", "hi"]) == 0

    def test_handler_invokes_terminate_tree(self):
        proc = MagicMock()
        proc.wait.return_value = 0
        proc.pid = 4242
        installed: dict[int, object] = {}

        def fake_signal(sig, handler):
            installed[sig] = handler

        with (
            patch.object(_reap.subprocess, "Popen", return_value=proc),
            patch.object(_reap.signal, "signal", side_effect=fake_signal),
            patch.object(_reap, "_terminate_tree") as tt,
        ):
            _reap._run_posix(["echo", "hi"])
            installed[signal.SIGTERM](signal.SIGTERM, None)

        tt.assert_called_once_with(os.getpid(), 4242)

    def test_keyboard_interrupt_triggers_terminate_tree(self):
        proc = MagicMock()
        proc.wait.side_effect = KeyboardInterrupt
        proc.pid = 9999
        with (
            patch.object(_reap.subprocess, "Popen", return_value=proc),
            patch.object(_reap.signal, "signal"),
            patch.object(_reap, "_terminate_tree") as tt,
        ):
            assert _reap._run_posix(["sleep", "9"]) == 130
        tt.assert_called_once_with(os.getpid(), 9999)


class TestMainDispatch:
    """``main`` dispatches to the platform-specific runner."""

    def test_routes_to_windows_runner_when_nt(self):
        with (
            patch.object(_reap.os, "name", "nt"),
            patch.object(_reap, "_run_windows", return_value=3) as rw,
            patch.object(_reap, "_run_posix") as rp,
        ):
            assert _reap.main(["--", "cmd", "arg"]) == 3
        rw.assert_called_once_with(["cmd", "arg"])
        rp.assert_not_called()

    def test_routes_to_posix_runner_otherwise(self):
        with (
            patch.object(_reap.os, "name", "posix"),
            patch.object(_reap, "_run_posix", return_value=5) as rp,
            patch.object(_reap, "_run_windows") as rw,
        ):
            assert _reap.main(["cmd", "arg"]) == 5
        rp.assert_called_once_with(["cmd", "arg"])
        rw.assert_not_called()


@pytest.mark.skipif(
    sys.platform == "win32", reason="module-main path only exercised on POSIX"
)
class TestModuleMain:
    """The ``__main__`` guard routes through ``main``."""

    def test_module_main_usage_error_exits_2(self):
        result = subprocess.run(
            [sys.executable, "-m", "marimo_jupyter_extension._reap"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "usage:" in result.stderr
