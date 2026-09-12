"""Private SentenceTransformer worker for the Cashew embedding boundary."""

from __future__ import annotations

import argparse
import json
import socket
import struct
from typing import Any

import numpy as np

_LENGTH = struct.Struct("!I")
_MAX_FRAME = 16 * 1024 * 1024
_MAX_DIMENSION = 65536


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive(sock: socket.socket) -> bytes:
    size = _LENGTH.unpack(_read_exact(sock, _LENGTH.size))[0]
    if size > _MAX_FRAME:
        raise ValueError("frame too large")
    return _read_exact(sock, size)


def _send(sock: socket.socket, payload: bytes) -> None:
    if len(payload) > _MAX_FRAME:
        raise ValueError("frame too large")
    sock.sendall(_LENGTH.pack(len(payload)) + payload)


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    effective_device = args.device
    kwargs = {} if args.device == "auto" else {"device": args.device}
    try:
        model = SentenceTransformer(args.model, **kwargs)
    except Exception:
        if args.device == "cpu":
            raise
        effective_device = "cpu"
        model = SentenceTransformer(args.model, device="cpu")
    if args.device == "auto" and getattr(model, "device", None) is not None:
        effective_device = str(model.device)

    get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    if get_dim is None:
        raise RuntimeError("embedding model has no dimension API")
    dimension = int(get_dim())
    if not 0 < dimension <= _MAX_DIMENSION:
        raise RuntimeError("invalid embedding dimension")
    if args.dimension > 0 and dimension != args.dimension:
        raise RuntimeError("embedding dimension mismatch")

    with socket.socket(fileno=args.fd) as channel:
        _send(
            channel,
            _json(
                {
                    "op": "hello",
                    "version": 1,
                    "generation": args.generation,
                    "model": args.model,
                    "requested_device": args.device,
                    "dimension": dimension,
                    "device": effective_device,
                }
            ),
        )
        while True:
            request = json.loads(_receive(channel))
            if request.get("generation") != args.generation:
                raise RuntimeError("generation mismatch")
            operation = request.get("op")
            if operation == "shutdown":
                return
            if (
                operation != "encode"
                or request.get("version") != 1
                or not isinstance(request.get("request_id"), str)
            ):
                raise RuntimeError("unsupported operation")
            texts = request.get("texts")
            if not isinstance(texts, list) or len(texts) > 100:
                raise RuntimeError("invalid text batch")
            if any(
                not isinstance(text, str) or len(text.encode("utf-8")) > 1048576
                for text in texts
            ):
                raise RuntimeError("invalid text")
            vectors = np.asarray(
                model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
                dtype="<f4",
            )
            if (
                vectors.shape != (len(texts), dimension)
                or not np.isfinite(vectors).all()
            ):
                raise RuntimeError("invalid model output")
            _send(
                channel,
                _json(
                    {
                        "op": "vectors",
                        "version": 1,
                        "generation": args.generation,
                        "request_id": request.get("request_id"),
                        "count": len(texts),
                        "dimension": dimension,
                        "device": effective_device,
                        "bytes": vectors.nbytes,
                    }
                ),
            )
            _send(channel, vectors.tobytes(order="C"))


if __name__ == "__main__":
    main()
