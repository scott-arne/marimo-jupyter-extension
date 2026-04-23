"""Tests for the ``_proxy_patch`` runtime patch on simpervisor."""

from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("simpervisor")

from simpervisor.process import SupervisedProcess  # noqa: E402

from marimo_jupyter_extension import _proxy_patch  # noqa: E402


@pytest.fixture
def fresh_patch():
    """Restore the original SupervisedProcess._signal_and_wait between tests."""
    original = SupervisedProcess._signal_and_wait
    if hasattr(SupervisedProcess, _proxy_patch._PATCHED_ATTR):
        delattr(SupervisedProcess, _proxy_patch._PATCHED_ATTR)
    yield
    SupervisedProcess._signal_and_wait = original
    if hasattr(SupervisedProcess, _proxy_patch._PATCHED_ATTR):
        delattr(SupervisedProcess, _proxy_patch._PATCHED_ATTR)


class _Fake:
    """Object with attributes the patched wrapper touches."""

    def __init__(self):
        self._killed = False
        self.running = True


class TestApply:
    """Patch installation."""

    def test_apply_reports_success(self, fresh_patch):
        assert _proxy_patch.apply() is True

    def test_apply_is_idempotent(self, fresh_patch):
        assert _proxy_patch.apply() is True
        assert _proxy_patch.apply() is True

    def test_apply_marks_class_as_patched(self, fresh_patch):
        _proxy_patch.apply()
        assert (
            getattr(SupervisedProcess, _proxy_patch._PATCHED_ATTR, False)
            is True
        )

    def test_apply_logs_success_when_logger_given(self, fresh_patch, caplog):
        log = logging.getLogger("test.apply.success")
        with caplog.at_level(logging.INFO, logger=log.name):
            _proxy_patch.apply(log)
        assert any(
            "guard installed" in rec.getMessage()
            for rec in caplog.records
            if rec.name == log.name
        )

    def test_apply_is_quiet_on_second_call(self, fresh_patch, caplog):
        log = logging.getLogger("test.apply.idempotent")
        _proxy_patch.apply(log)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=log.name):
            _proxy_patch.apply(log)
        assert not [rec for rec in caplog.records if rec.name == log.name]

    def test_missing_attributes_degrade_gracefully(self, fresh_patch):
        """If simpervisor renames ``_killed`` / ``running``, the wrapper
        must not create phantom attributes — it must swallow the lookup
        error silently."""

        async def raising(self, signum):  # noqa: ARG001
            raise ProcessLookupError()

        SupervisedProcess._signal_and_wait = raising
        _proxy_patch.apply()

        class Bare:
            pass

        bare = Bare()
        result = asyncio.run(SupervisedProcess._signal_and_wait(bare, 15))

        assert result is None
        assert not hasattr(bare, "_killed")
        assert not hasattr(bare, "running")


class TestPatchedBehavior:
    """The wrapped ``_signal_and_wait`` swallows ProcessLookupError only."""

    def test_process_lookup_error_is_swallowed(self, fresh_patch):
        async def raising(self, signum):  # noqa: ARG001
            raise ProcessLookupError()

        SupervisedProcess._signal_and_wait = raising
        _proxy_patch.apply()

        fake = _Fake()
        result = asyncio.run(SupervisedProcess._signal_and_wait(fake, 15))

        assert result is None
        assert fake._killed is True
        assert fake.running is False

    def test_successful_signal_passes_through(self, fresh_patch):
        calls: list[int] = []

        async def success(self, signum):  # noqa: ARG001
            calls.append(signum)
            return "ok"

        SupervisedProcess._signal_and_wait = success
        _proxy_patch.apply()

        fake = _Fake()
        result = asyncio.run(SupervisedProcess._signal_and_wait(fake, 15))

        assert result == "ok"
        assert calls == [15]
        assert fake._killed is False
        assert fake.running is True

    def test_other_exceptions_still_propagate(self, fresh_patch):
        async def boom(self, signum):  # noqa: ARG001
            raise RuntimeError("boom")

        SupervisedProcess._signal_and_wait = boom
        _proxy_patch.apply()

        fake = _Fake()
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(SupervisedProcess._signal_and_wait(fake, 15))


class TestDegradedPaths:
    """Import and attribute failures return False and log a warning."""

    def test_returns_false_when_simpervisor_not_importable(self, caplog):
        """If ``simpervisor`` cannot be imported the patch reports failure
        and writes a warning to the supplied logger."""
        log = logging.getLogger("test.apply.no.simpervisor")
        # Remove the already-imported simpervisor modules so that the fresh
        # import inside ``apply`` re-runs and can be forced to fail.
        saved = {
            name: mod
            for name, mod in list(sys.modules.items())
            if name == "simpervisor" or name.startswith("simpervisor.")
        }
        for name in saved:
            del sys.modules[name]
        try:
            original_import = (
                __builtins__["__import__"]
                if isinstance(__builtins__, dict)
                else __builtins__.__import__
            )

            def fake_import(name, *args, **kwargs):
                if name == "simpervisor.process" or name.startswith(
                    "simpervisor"
                ):
                    raise ImportError("forced")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                with caplog.at_level(logging.WARNING, logger=log.name):
                    assert _proxy_patch.apply(log) is False
        finally:
            sys.modules.update(saved)

        assert any(
            "simpervisor not importable" in rec.getMessage()
            for rec in caplog.records
            if rec.name == log.name
        )

    def test_returns_false_when_signal_method_missing(
        self, fresh_patch, caplog
    ):
        """If ``simpervisor`` renames ``_signal_and_wait`` the patch refuses
        to install and logs a warning."""
        log = logging.getLogger("test.apply.no.method")
        fake_proc_module = MagicMock()

        class _NoSignalSupervisedProcess:
            pass

        fake_proc_module.SupervisedProcess = _NoSignalSupervisedProcess
        with patch.dict(
            sys.modules, {"simpervisor.process": fake_proc_module}
        ):
            with caplog.at_level(logging.WARNING, logger=log.name):
                assert _proxy_patch.apply(log) is False

        assert any(
            "_signal_and_wait not found" in rec.getMessage()
            for rec in caplog.records
            if rec.name == log.name
        )
