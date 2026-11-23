from contextlib import contextmanager
from socket import AF_INET, SOCK_STREAM, socket
from threading import Thread
from time import monotonic, sleep

import pytest
import uvicorn


@pytest.fixture(scope="module")
def serve_app():
    @contextmanager
    def serve(app):
        with socket(AF_INET, SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    log_config=None,
                    access_log=False,
                    ws="websockets-sansio",
                    timeout_graceful_shutdown=2,
                )
            )
            thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
            thread.start()
            try:
                deadline = monotonic() + 5
                while not server.started:
                    if not thread.is_alive() or monotonic() >= deadline:
                        raise RuntimeError("Test server failed to start")
                    sleep(0.01)
                yield f"http://127.0.0.1:{port}"
            finally:
                server.should_exit = True
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("Test server failed to shut down")

    return serve
