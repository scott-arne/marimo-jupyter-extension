"""In-process tests for ``handlers.py``.

These tests exercise the handler classes and the
``_load_jupyter_server_extension`` hook directly via mocks so that the
module's executable lines register with the coverage collector. We invoke
handlers via ``asyncio.run`` rather than ``pytest-asyncio`` to avoid adding
a new test dependency.
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("jupyter_server")

from marimo_jupyter_extension import handlers  # noqa: E402


def _make_handler(handler_cls, *, body=b"{}", user="u", application=None):
    """Build a handler instance bypassing Tornado's initializer."""
    handler = handler_cls.__new__(handler_cls)
    handler.request = SimpleNamespace(body=body)
    handler.application = application
    handler.current_user = user
    handler.set_status = MagicMock()
    handler.finish = MagicMock()
    return handler


def _run(handler, method_name):
    """Invoke the ``@web.authenticated`` async method, bypassing the guard."""
    method = getattr(type(handler), method_name).__wrapped__
    asyncio.run(method(handler))


class TestFindMarimoProxyState:
    """``_find_marimo_proxy_state`` scans registered handlers."""

    def _spec(self, pattern, state=None):
        spec = SimpleNamespace()
        spec.regex = re.compile(pattern)
        spec.kwargs = {"state": state} if state is not None else {}
        return spec

    def test_returns_state_for_marimo_pattern(self):
        state = {"proc": "x"}
        web_app = SimpleNamespace(
            handlers=[
                (".*", [self._spec(r"/other", state={"x": 1})]),
                (".*", [self._spec(r"/marimo/edit", state=state)]),
            ]
        )
        assert handlers._find_marimo_proxy_state(web_app) is state

    def test_returns_none_when_no_match(self):
        web_app = SimpleNamespace(
            handlers=[(".*", [self._spec(r"/other", state={"x": 1})])]
        )
        assert handlers._find_marimo_proxy_state(web_app) is None

    def test_ignores_specs_without_state_kwarg(self):
        web_app = SimpleNamespace(handlers=[(".*", [self._spec(r"/marimo")])])
        assert handlers._find_marimo_proxy_state(web_app) is None


class TestConvertHandler:
    """``ConvertHandler.post`` validates input and delegates to convert."""

    def test_missing_input_returns_400(self):
        handler = _make_handler(
            handlers.ConvertHandler,
            body=json.dumps({"output": "x.py"}).encode(),
        )
        _run(handler, "post")
        handler.set_status.assert_called_once_with(400)
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is False
        assert "Missing" in payload["error"]

    def test_missing_output_returns_400(self):
        handler = _make_handler(
            handlers.ConvertHandler,
            body=json.dumps({"input": "x.ipynb"}).encode(),
        )
        _run(handler, "post")
        handler.set_status.assert_called_once_with(400)

    def test_calls_convert_and_finishes_with_success(self):
        with patch.object(handlers, "convert_notebook_to_marimo") as cv:
            handler = _make_handler(
                handlers.ConvertHandler,
                body=json.dumps(
                    {"input": "x.ipynb", "output": "x.py"}
                ).encode(),
            )
            _run(handler, "post")
        cv.assert_called_once_with("x.ipynb", "x.py")
        payload = handler.finish.call_args[0][0]
        assert payload == {"success": True, "output": "x.py"}

    def test_runtime_error_returns_500(self):
        with patch.object(
            handlers,
            "convert_notebook_to_marimo",
            side_effect=RuntimeError("conv failed"),
        ):
            handler = _make_handler(
                handlers.ConvertHandler,
                body=json.dumps(
                    {"input": "x.ipynb", "output": "x.py"}
                ).encode(),
            )
            _run(handler, "post")
        handler.set_status.assert_called_once_with(500)
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is False
        assert "conv failed" in payload["error"]


class TestRestartHandler:
    """``RestartHandler.post`` kills and clears the cached proxy process."""

    def _state(self, proc):
        lock = MagicMock()
        lock.__aenter__ = AsyncMock(return_value=None)
        lock.__aexit__ = AsyncMock(return_value=None)
        return {"proc": proc, "proc_lock": lock}

    def test_returns_503_when_state_missing(self):
        with patch.object(
            handlers, "_find_marimo_proxy_state", return_value=None
        ):
            handler = _make_handler(
                handlers.RestartHandler, application=SimpleNamespace()
            )
            _run(handler, "post")
        handler.set_status.assert_called_once_with(503)
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is False

    def test_kills_live_process_and_clears_state(self):
        proc = MagicMock()
        proc.kill = AsyncMock()
        state = self._state(proc)
        with patch.object(
            handlers, "_find_marimo_proxy_state", return_value=state
        ):
            handler = _make_handler(
                handlers.RestartHandler, application=SimpleNamespace()
            )
            _run(handler, "post")
        proc.kill.assert_awaited_once()
        assert "proc" not in state
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is True

    def test_swallows_kill_exception(self):
        proc = MagicMock()
        proc.kill = AsyncMock(side_effect=RuntimeError("already dead"))
        state = self._state(proc)
        with patch.object(
            handlers, "_find_marimo_proxy_state", return_value=state
        ):
            handler = _make_handler(
                handlers.RestartHandler, application=SimpleNamespace()
            )
            _run(handler, "post")
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is True
        assert "proc" not in state

    def test_skips_kill_when_process_not_managed(self):
        state = self._state("process not managed")
        with patch.object(
            handlers, "_find_marimo_proxy_state", return_value=state
        ):
            handler = _make_handler(
                handlers.RestartHandler, application=SimpleNamespace()
            )
            _run(handler, "post")
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is True


class TestConfigHandler:
    """``ConfigHandler.get`` returns the extension config."""

    def test_returns_no_sandbox_flag(self):
        fake_config = SimpleNamespace(no_sandbox=True)
        with patch(
            "marimo_jupyter_extension.config.get_config",
            return_value=fake_config,
        ):
            handler = _make_handler(handlers.ConfigHandler)
            _run(handler, "get")
        handler.finish.assert_called_once_with({"no_sandbox": True})


class TestCreateStubHandler:
    """``CreateStubHandler.post`` writes a stub file with optional PEP 723."""

    def test_missing_path_returns_400(self):
        handler = _make_handler(
            handlers.CreateStubHandler,
            body=json.dumps({}).encode(),
        )
        _run(handler, "post")
        handler.set_status.assert_called_once_with(400)

    def test_writes_stub_without_venv(self, tmp_path):
        target = tmp_path / "stub.py"
        handler = _make_handler(
            handlers.CreateStubHandler,
            body=json.dumps({"path": str(target)}).encode(),
        )
        _run(handler, "post")
        handler.finish.assert_called_once()
        content = target.read_text()
        assert "import marimo" in content
        assert "# /// script" not in content

    def test_writes_stub_with_venv_header(self, tmp_path):
        target = tmp_path / "stub.py"
        venv_python = tmp_path / "venv" / "bin" / "python3.12"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        handler = _make_handler(
            handlers.CreateStubHandler,
            body=json.dumps(
                {"path": str(target), "venv": str(venv_python)}
            ).encode(),
        )
        _run(handler, "post")
        content = target.read_text()
        assert "# /// script" in content
        assert str(tmp_path / "venv") in content

    def test_handles_venv_path_without_bin(self, tmp_path):
        target = tmp_path / "stub.py"
        venv_python = tmp_path / "python"
        venv_python.touch()
        handler = _make_handler(
            handlers.CreateStubHandler,
            body=json.dumps(
                {"path": str(target), "venv": str(venv_python)}
            ).encode(),
        )
        _run(handler, "post")
        content = target.read_text()
        assert "# /// script" in content

    def test_write_failure_returns_500(self, tmp_path):
        bad_path = tmp_path / "no-such-dir" / "stub.py"
        handler = _make_handler(
            handlers.CreateStubHandler,
            body=json.dumps({"path": str(bad_path)}).encode(),
        )
        _run(handler, "post")
        handler.set_status.assert_called_once_with(500)
        payload = handler.finish.call_args[0][0]
        assert payload["success"] is False


class TestExtensionHooks:
    """Server-extension entry points."""

    def test_extension_points_returns_this_module(self):
        result = handlers._jupyter_server_extension_points()
        assert result == [{"module": "marimo_jupyter_extension.handlers"}]

    def test_load_extension_registers_handlers_and_applies_patch(self):
        added = []
        web_app = MagicMock()
        web_app.settings = {"base_url": "/"}
        web_app.add_handlers = MagicMock(
            side_effect=lambda host, items: added.extend(items)
        )
        server_app = SimpleNamespace(web_app=web_app, log=MagicMock())

        with patch(
            "marimo_jupyter_extension._proxy_patch.apply"
        ) as apply_mock:
            handlers._load_jupyter_server_extension(server_app)

        apply_mock.assert_called_once_with(server_app.log)
        routes = [route for route, _cls in added]
        assert any("convert" in r for r in routes)
        assert any("restart" in r for r in routes)
        assert any("create-stub" in r for r in routes)
        assert any("config" in r for r in routes)
        server_app.log.info.assert_called_once()
