# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the persistent session connection against a WebSocket server."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.server import serve

from nemo_fabric_adapters.common.lifecycle import LifecycleError
from nemo_fabric_adapters.openclaw import gateway


async def test_session_calls_share_one_authenticated_connection(
    unused_tcp_port, caplog
):
    caplog.set_level("DEBUG", logger="websockets.client")
    connections = []
    requests = []

    async def respond(socket):
        connections.append(socket)
        await socket.send(
            json.dumps(
                {
                    "type": "event",
                    "event": "connect.challenge",
                    "payload": {"nonce": "challenge"},
                }
            )
        )
        async for message in socket:
            request = json.loads(message)
            requests.append(request)
            if request["method"] == "connect":
                assert request["params"]["auth"] == {"token": "private-test-token"}
                assert request["params"]["client"]["mode"] == "backend"
                result = {"type": "hello-ok", "protocol": 4}
            else:
                result = {
                    "method": request["method"],
                    "history": "private-test-history",
                }
            await socket.send(json.dumps({"type": "event", "event": "tick"}))
            await socket.send(
                json.dumps(
                    {"type": "res", "id": request["id"], "ok": True, "payload": result}
                )
            )

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        try:
            for _ in range(7):
                assert await client.call(
                    "sessions.describe", {"key": "selected"}, timeout=2
                ) == {"method": "sessions.describe", "history": "private-test-history"}
        finally:
            await client.close()
        assert "private-test-token" not in caplog.text
        assert "private-test-history" not in caplog.text
        assert len(connections) == 1
        assert len(requests) == 8
        await asyncio.wait_for(connections[0].wait_closed(), 2)


async def _serve_with_event_bursts(socket, *, burst: int, requests: list[dict]):
    """Answer RPCs, then flood the idle connection with unsolicited events."""
    await socket.send(
        json.dumps(
            {
                "type": "event",
                "event": "connect.challenge",
                "payload": {"nonce": "challenge"},
            }
        )
    )
    async for message in socket:
        request = json.loads(message)
        requests.append(request)
        payload = (
            {"type": "hello-ok", "protocol": 4}
            if request["method"] == "connect"
            else {"method": request["method"], "key": request["params"]["key"]}
        )
        await socket.send(
            json.dumps(
                {"type": "res", "id": request["id"], "ok": True, "payload": payload}
            )
        )
        if request["method"] != "connect":
            for index in range(burst):
                await socket.send(
                    json.dumps(
                        {"type": "event", "event": "agent", "payload": {"n": index}}
                    )
                )


async def test_session_stays_responsive_while_idle_between_calls(unused_tcp_port):
    # The Gateway keeps emitting events while a native turn runs over HTTP and no
    # RPC is waiting on this connection. Unread events must not stall keepalive.
    requests: list[dict] = []

    async def respond(socket):
        await _serve_with_event_bursts(socket, burst=200, requests=requests)

    async with serve(
        respond, "127.0.0.1", unused_tcp_port, ping_interval=0.2, ping_timeout=0.5
    ):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        try:
            assert await client.call("sessions.describe", {"key": "first"}, timeout=2)
            await asyncio.sleep(1.5)
            assert await client.call(
                "sessions.describe", {"key": "second"}, timeout=2
            ) == {"method": "sessions.describe", "key": "second"}
        finally:
            await client.close()
        assert [request["method"] for request in requests] == [
            "connect",
            "sessions.describe",
            "sessions.describe",
        ]


async def test_session_events_before_a_response_are_discarded(unused_tcp_port):
    async def respond(socket):
        await socket.send(
            json.dumps(
                {
                    "type": "event",
                    "event": "connect.challenge",
                    "payload": {"nonce": "challenge"},
                }
            )
        )
        async for message in socket:
            request = json.loads(message)
            for index in range(64):
                await socket.send(
                    json.dumps({"type": "event", "event": "agent", "payload": index})
                )
            payload = (
                {"type": "hello-ok", "protocol": 4}
                if request["method"] == "connect"
                else {"method": request["method"]}
            )
            await socket.send(
                json.dumps(
                    {
                        "type": "res",
                        "id": request["id"],
                        "ok": True,
                        "payload": payload,
                    }
                )
            )

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        try:
            for method in ("sessions.describe", "sessions.history", "sessions.abort"):
                assert await client.call(method, {"key": "selected"}, timeout=2) == {
                    "method": method
                }
        finally:
            await client.close()


async def test_session_close_stops_the_receiver(unused_tcp_port):
    requests: list[dict] = []

    def client_tasks() -> list[asyncio.Task]:
        return [
            task
            for task in asyncio.all_tasks()
            if getattr(task.get_coro(), "__qualname__", "").startswith("SessionClient.")
            and task is not asyncio.current_task()
        ]

    async def respond(socket):
        await _serve_with_event_bursts(socket, burst=3, requests=requests)

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        assert await client.call("sessions.describe", {"key": "selected"}, timeout=2)
        assert len(client_tasks()) == 1
        await client.close()
        assert client_tasks() == []
        with pytest.raises(LifecycleError):
            await client.call("sessions.describe", {"key": "selected"}, timeout=2)


async def test_closing_the_client_releases_a_waiting_call(unused_tcp_port):
    received = asyncio.Event()

    async def respond(socket):
        await socket.send(
            json.dumps(
                {
                    "type": "event",
                    "event": "connect.challenge",
                    "payload": {"nonce": "challenge"},
                }
            )
        )
        async for message in socket:
            request = json.loads(message)
            if request["method"] == "connect":
                await socket.send(
                    json.dumps(
                        {
                            "type": "res",
                            "id": request["id"],
                            "ok": True,
                            "payload": {"type": "hello-ok", "protocol": 4},
                        }
                    )
                )
            else:
                received.set()

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        waiting = asyncio.create_task(
            client.call("sessions.describe", {"key": "selected"}, timeout=30)
        )
        await asyncio.wait_for(received.wait(), 2)
        await client.close()
        with pytest.raises(LifecycleError):
            await asyncio.wait_for(waiting, 2)


@pytest.mark.parametrize(
    "failure",
    [
        "authentication",
        "rpc",
        "timeout",
        "cancel",
        "disconnect",
        "invalid",
        "unknown_response",
    ],
)
async def test_session_failures_close_connection_without_replay(
    unused_tcp_port, failure
):
    requests = []
    received = asyncio.Event()

    async def respond(socket):
        await socket.send(
            json.dumps(
                {
                    "type": "event",
                    "event": "connect.challenge",
                    "payload": {"nonce": "challenge"},
                }
            )
        )
        async for message in socket:
            request = json.loads(message)
            requests.append(request["method"])
            if request["method"] == "connect":
                await socket.send(
                    json.dumps(
                        {
                            "type": "res",
                            "id": request["id"],
                            "ok": failure != "authentication",
                            "payload": {"type": "hello-ok", "protocol": 4},
                            "error": {"message": "secret-server-detail"},
                        }
                    )
                )
                continue
            received.set()
            if failure in {"timeout", "cancel"}:
                await socket.wait_closed()
            elif failure == "disconnect":
                await socket.close()
            elif failure == "invalid":
                await socket.send("not-json-secret-server-detail")
            elif failure == "unknown_response":
                await socket.send(
                    json.dumps(
                        {"type": "res", "id": "unrelated", "ok": True, "payload": {}}
                    )
                )
            else:
                await socket.send(
                    json.dumps(
                        {
                            "type": "res",
                            "id": request["id"],
                            "ok": False,
                            "error": {"message": "secret-server-detail"},
                        }
                    )
                )

    async with serve(respond, "127.0.0.1", unused_tcp_port):
        client = gateway.SessionClient(
            f"http://127.0.0.1:{unused_tcp_port}", "private-test-token"
        )
        try:
            task = asyncio.create_task(
                client.call(
                    "sessions.abort",
                    {"key": "selected"},
                    timeout=0.1 if failure == "timeout" else 2,
                )
            )
            if failure == "cancel":
                await asyncio.wait_for(received.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(LifecycleError) as caught:
                    await task
                assert "secret-server-detail" not in str(caught.value)
                assert "private-test-token" not in str(caught.value)
            with pytest.raises(LifecycleError):
                await client.call("sessions.abort", {"key": "selected"}, timeout=2)
            assert requests.count("connect") == 1
            assert requests.count("sessions.abort") <= 1
        finally:
            await client.close()


async def test_runtime_session_checks_do_not_spawn_cli_processes():
    from nemo_fabric_adapters.openclaw import adapter

    runtime = adapter.OpenClawRuntime()
    runtime._session_client = AsyncMock(spec=gateway.SessionClient)
    runtime._session_client.call.return_value = {"session": None}
    try:
        assert await runtime._session_call(
            "sessions.describe", {"key": "selected"}
        ) == {"session": None}
        runtime._session_client.call.assert_awaited_once()
    finally:
        await runtime.stop()


async def test_gateway_redirect_never_receives_authentication(unused_tcp_port_factory):
    original_port = unused_tcp_port_factory()
    redirected_port = unused_tcp_port_factory()
    redirected = []

    async def observe(socket):
        redirected.append(socket)
        await socket.close()

    async def redirect(connection, request):
        response = connection.respond(302, "Moved")
        response.headers["Location"] = f"ws://127.0.0.1:{redirected_port}/"
        return response

    async with serve(observe, "127.0.0.1", redirected_port):
        async with serve(observe, "127.0.0.1", original_port, process_request=redirect):
            client = gateway.SessionClient(
                f"http://127.0.0.1:{original_port}", "private-test-token"
            )
            try:
                with pytest.raises(LifecycleError):
                    await client.call(
                        "sessions.describe", {"key": "selected"}, timeout=2
                    )
                assert redirected == []
            finally:
                await client.close()
