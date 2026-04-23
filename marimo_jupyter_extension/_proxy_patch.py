"""Runtime patch for a ``simpervisor`` race that breaks marimo respawn.

When marimo exits on its own (for example on ``--timeout`` idle shutdown),
``jupyter-server-proxy`` still holds a cached ``SupervisedProcess`` whose
``asyncio.subprocess`` child has already been reaped. On the next request
the proxy's ``ensure_process`` tries to ``await proc.kill()`` as part of
its failed-to-ready cleanup, which ultimately calls
``asyncio.base_subprocess.BaseSubprocessTransport.send_signal`` — and that
raises ``ProcessLookupError`` because the transport knows the process is
gone.

Rather than wrapping the proxy handler, this module patches the single
call site inside ``simpervisor.process.SupervisedProcess._signal_and_wait``
so that both ``terminate()`` and ``kill()`` treat "already dead" as
success. The monkey-patch is idempotent and is applied from
``_load_jupyter_server_extension``.
"""

from __future__ import annotations

import logging
from typing import Any

_PATCHED_ATTR = "_marimo_jupyter_extension_patched"

_SILENT_LOG = logging.getLogger(__name__)


def apply(log: Any = None) -> bool:
    """Install the ``send_signal`` guard on ``SupervisedProcess``.

    :param log: Optional logger (``logging.Logger`` or Traitlets logger)
        used to report patch status. If omitted, a module-level logger
        is used so failures are still recorded somewhere.
    :returns: True if the patch was applied (or was already applied);
        False if ``simpervisor`` could not be imported or the expected
        method is missing (e.g. an incompatible future version).
    """
    log = log if log is not None else _SILENT_LOG

    try:
        from simpervisor.process import SupervisedProcess
    except ImportError:
        log.warning(
            "marimo-jupyter-extension: simpervisor not importable; "
            "SupervisedProcess ProcessLookupError guard NOT installed. "
            "Marimo respawn after idle-exit may return 500 errors."
        )
        return False

    if getattr(SupervisedProcess, _PATCHED_ATTR, False):
        return True

    original = getattr(SupervisedProcess, "_signal_and_wait", None)
    if original is None:
        log.warning(
            "marimo-jupyter-extension: "
            "simpervisor.SupervisedProcess._signal_and_wait not found "
            "(incompatible simpervisor version?); guard NOT installed."
        )
        return False

    async def _signal_and_wait(self, signum):
        try:
            return await original(self, signum)
        except ProcessLookupError:
            # The child was already reaped. Mark the supervised process as
            # killed so subsequent terminate()/kill() calls don't raise
            # KilledProcessError from a stale handle, and return cleanly.
            # Guard the attribute writes so that a future simpervisor
            # refactor that renames these fields degrades to a warning
            # rather than creating phantom attributes.
            if hasattr(self, "_killed"):
                self._killed = True
            if hasattr(self, "running"):
                self.running = False
            return None

    SupervisedProcess._signal_and_wait = _signal_and_wait
    setattr(SupervisedProcess, _PATCHED_ATTR, True)
    log.info(
        "marimo-jupyter-extension: "
        "SupervisedProcess ProcessLookupError guard installed."
    )
    return True
