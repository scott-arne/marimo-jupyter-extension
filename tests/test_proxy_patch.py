"""Tests for the ``_proxy_patch`` runtime patch on simpervisor."""

from __future__ import annotations

import asyncio
import logging

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
