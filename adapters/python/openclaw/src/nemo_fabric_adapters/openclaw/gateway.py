# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One authenticated Gateway connection for a runtime's native session checks."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidHandshake, WebSocketException

from nemo_fabric_adapters.common.lifecycle import LifecycleError


# Gateway frames contain credentials, conversation history, and tool arguments.
# Keep this unregistered logger private so global DEBUG settings cannot emit frames.
_TRANSPORT_LOGGER = logging.Logger(__name__ + ".transport")
_TRANSPORT_LOGGER.disabled = True


class _DirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        # Connection material is valid only for the selected Gateway.
        return InvalidHandshake("Gateway redirects are unsupported")


class SessionClient:
    """Serialize native session RPCs over one connection a receiver keeps drained.

    The Gateway streams events for as long as a native turn runs, including while
    no session RPC is waiting. A receiver task reads every frame, resolves the
    response each RPC awaits, and discards events, so unread frames never fill the
    socket's receive queue or stall WebSocket keepalive.
    """

    def __init__(self, gateway_url: str, token: str):
        parsed = urlsplit(gateway_url)
        self._url = urlunsplit(
            ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, "/", "", "")
        )
        self._token = token
        self._connection: ClientConnection | None = None
        self._receiver: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._failed = False

    async def _receive(self, connection: ClientConnection) -> None:
        failure: BaseException = OSError("Gateway receiver stopped")
        try:
            while True:
                frame = json.loads(await connection.recv())
                if not isinstance(frame, dict):
                    raise ValueError("Invalid Gateway frame")
                if frame.get("type") == "event":
                    continue
                identifier = frame.get("id")
                pending = (
                    self._pending.get(identifier)
                    if frame.get("type") == "res" and isinstance(identifier, str)
                    else None
                )
                if pending is None:
                    raise ValueError("Unexpected Gateway frame")
                if pending.done():
                    continue
                if frame.get("ok") is not True or not isinstance(
                    frame.get("payload"), dict
                ):
                    pending.set_exception(
                        ValueError("Gateway operation was not confirmed")
                    )
                else:
                    pending.set_result(frame["payload"])
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = error
            self._failed = True
            await connection.close()
        finally:
            for pending in self._pending.values():
                if not pending.done():
                    pending.set_exception(failure)

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self._connection is not None
        assert self._receiver is not None
        identifier = str(uuid4())
        pending: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[identifier] = pending
        try:
            if self._receiver.done():
                raise OSError("Gateway receiver stopped")
            await self._connection.send(
                json.dumps(
                    {
                        "type": "req",
                        "id": identifier,
                        "method": method,
                        "params": params,
                    }
                )
            )
            return await pending
        finally:
            del self._pending[identifier]
            if pending.done() and not pending.cancelled():
                pending.exception()  # Mark a failure nobody awaited as retrieved.

    async def call(
        self, method: str, params: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        async with self._lock:
            if self._failed:
                raise LifecycleError(
                    "session_incompatible", "Native session connection is unavailable"
                )
            try:
                async with asyncio.timeout(timeout):
                    if self._connection is None:
                        self._connection = await _DirectConnect(
                            self._url,
                            proxy=None,
                            logger=_TRANSPORT_LOGGER,
                            open_timeout=timeout,
                            close_timeout=0.25,
                            max_size=16 * 1024 * 1024,
                        )
                        challenge = json.loads(await self._connection.recv())
                        if (
                            not isinstance(challenge, dict)
                            or challenge.get("type") != "event"
                            or challenge.get("event") != "connect.challenge"
                        ):
                            raise ValueError("Gateway challenge is unavailable")
                        self._receiver = asyncio.create_task(
                            self._receive(self._connection)
                        )
                        hello = await self._request(
                            "connect",
                            {
                                "minProtocol": 4,
                                "maxProtocol": 4,
                                "client": {
                                    "id": "gateway-client",
                                    "displayName": "NeMo Fabric",
                                    "version": "1",
                                    "platform": sys.platform,
                                    "mode": "backend",
                                },
                                "role": "operator",
                                "scopes": ["operator.admin"],
                                "auth": {"token": self._token},
                            },
                        )
                        if (
                            hello.get("type") != "hello-ok"
                            or hello.get("protocol") != 4
                        ):
                            raise ValueError("Unsupported Gateway protocol")
                    return await self._request(method, params)
            except BaseException as error:
                await self.close()
                if isinstance(
                    error, (OSError, ValueError, TimeoutError, WebSocketException)
                ):
                    # Native errors can contain prompts or authentication material.
                    raise LifecycleError(
                        "session_incompatible",
                        "Native session operation could not be confirmed",
                        metadata={"method": method},
                    ) from None
                raise

    async def close(self) -> None:
        self._failed = True
        receiver, self._receiver = self._receiver, None
        connection, self._connection = self._connection, None
        if receiver is not None:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        if connection is not None:
            await connection.close()
