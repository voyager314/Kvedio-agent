import json

import httpx
import pytest

from kvedio_agent1.core.domain import UsageKind
from kvedio_agent1.core.llm import Embedder, ModelError

URL = "https://embed.example.invalid/v1/embeddings"
KEY = "sk-live-do-not-print"


def build(handler, **options) -> Embedder:
    return Embedder(
        url=URL,
        api_key=KEY,
        model="embed-model",
        timeout=5,
        transport=httpx.MockTransport(handler),
        **options,
    )


def reply(vectors: list[list[float]], *, indices: list[int] | None = None) -> httpx.Response:
    data = [
        {"index": (indices or list(range(len(vectors))))[position], "embedding": vector}
        for position, vector in enumerate(vectors)
    ]
    return httpx.Response(200, json={"data": data, "model": "embed-model"})


def test_the_request_uses_the_configured_address_and_model():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.read())
        return reply([[1.0, 0.0]])

    build(handler).embed(["预算"])
    assert seen["url"] == URL
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"] == {"model": "embed-model", "input": ["预算"]}


def test_vectors_come_back_in_the_order_of_the_texts():
    def handler(request: httpx.Request) -> httpx.Response:
        return reply([[0.1, 0.2], [0.3, 0.4]], indices=[1, 0])

    assert build(handler).embed(["甲", "乙"]) == [[0.3, 0.4], [0.1, 0.2]]


def test_long_input_is_sent_in_batches():
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        batch = json.loads(request.read())["input"]
        calls.append(batch)
        return reply([[float(len(text))] for text in batch])

    vectors = build(handler, batch_size=2).embed(["一", "二", "三"])
    assert [len(batch) for batch in calls] == [2, 1]
    assert [vector[0] for vector in vectors] == [1.0, 1.0, 1.0]


def test_no_texts_means_no_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("不应发出请求")

    assert build(handler).embed([]) == []


def test_a_server_fault_is_retried():
    calls: list[int] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return reply([[0.5, 0.5]])

    vectors = build(handler, sleep=delays.append).embed(["预算"])
    assert vectors == [[0.5, 0.5]]
    assert len(calls) == 2
    assert delays == [1.0]


def test_a_count_mismatch_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return reply([[0.1, 0.2]])

    with pytest.raises(ModelError, match="条数"):
        build(handler).embed(["甲", "乙"])


def test_a_missing_vector_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0}]})

    with pytest.raises(ModelError, match="向量"):
        build(handler).embed(["甲"])


def test_inconsistent_dimensions_are_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return reply([[0.1, 0.2], [0.3]])

    with pytest.raises(ModelError, match="维度"):
        build(handler).embed(["甲", "乙"])


def test_the_key_never_appears_in_an_error_or_the_repr():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    embedder = build(handler)
    assert KEY not in repr(embedder)
    assert URL in repr(embedder)
    with pytest.raises(ModelError) as failure:
        embedder.embed(["甲"])
    assert KEY not in str(failure.value)

def test_an_embedding_call_reports_its_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 9},
            },
        )

    seen = []
    build(handler, usage=seen.append).embed(["甲"])

    assert len(seen) == 1
    assert seen[0].kind is UsageKind.EMBEDDING
    assert seen[0].prompt_tokens == 9
