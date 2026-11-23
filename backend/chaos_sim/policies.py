"""Resolve configured policy routes and check their HTTP health."""

from collections.abc import Mapping
from os import getenv

import httpx
from pydantic import TypeAdapter

from backend.shared.schemas import ModelId, PolicyHealthResponse

DEFAULT_POLICY_URL = "http://127.0.0.1:8001"
MODEL_ID = TypeAdapter(ModelId)
HEALTH_TIMEOUT = httpx.Timeout(2.0, connect=1.0)


def parse_policy_servers(value: str) -> dict[str, str]:
    servers = {}
    for entry in value.split(","):
        name, separator, url = (part.strip() for part in entry.partition("="))
        if not separator or name in servers:
            raise ValueError(f"Expected a unique model=url entry in POLICY_SERVERS, got {entry!r}")
        servers[name] = url
    return normalize_policy_servers(servers)


def normalize_policy_servers(servers: Mapping[str, str]) -> dict[str, str]:
    if not servers:
        raise ValueError("At least one policy server is required")
    normalized = {}
    for name, address in servers.items():
        MODEL_ID.validate_python(name)
        try:
            url = httpx.URL(address)
        except httpx.InvalidURL as error:
            raise ValueError(f"Invalid HTTP URL for policy {name!r}") from error
        if url.scheme not in ("http", "https") or not url.host or url.query or url.fragment:
            raise ValueError(f"Invalid HTTP base URL for policy {name!r}")
        normalized[name] = str(url).rstrip("/")
    return normalized


def configured_policy_servers(
    policy_url: str | None, policy_servers: Mapping[str, str] | None
) -> dict[str, str]:
    if policy_servers is not None:
        if policy_url is not None:
            raise ValueError("Pass either policy_url or policy_servers")
        return normalize_policy_servers(policy_servers)
    if policy_url is None and (registry := getenv("POLICY_SERVERS")):
        return parse_policy_servers(registry)
    return normalize_policy_servers(
        {"default": policy_url or getenv("POLICY_SERVER_URL", DEFAULT_POLICY_URL)}
    )


async def fetch_policy_health(client: httpx.AsyncClient, url: str) -> PolicyHealthResponse:
    response = await client.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
    response.raise_for_status()
    return PolicyHealthResponse.model_validate_json(response.content)
