import pytest

from schemas import Request

from utils.block import (
    build_logical_blocks,
    append_decode_token,
    compute_block_hash,
)


BLOCK_SIZE = 4


def test_build_logical_blocks_all_full():
    request = Request(
        request_id="0",
        input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    build_logical_blocks(request, BLOCK_SIZE)

    assert len(request.logical_blocks) == 2

    block0 = request.logical_blocks[0]
    block1 = request.logical_blocks[1]

    assert block0.token_ids == [1, 2, 3, 4]
    assert block1.token_ids == [5, 6, 7, 8]

    assert block0.block_hash == compute_block_hash(None, [1, 2, 3, 4])
    assert block1.block_hash == compute_block_hash(
        block0.block_hash,
        [5, 6, 7, 8],
    )


def test_build_logical_blocks_last_not_full():
    request = Request(
        request_id="0",
        input_ids=[1, 2, 3, 4, 5],
    )

    build_logical_blocks(request, BLOCK_SIZE)

    assert len(request.logical_blocks) == 2

    assert request.logical_blocks[0].block_hash is not None
    assert request.logical_blocks[1].block_hash is None

    assert request.logical_blocks[1].token_ids == [5]


def test_append_to_partial_block():
    request = Request(
        request_id="0",
        input_ids=[1, 2, 3, 4, 5],
    )

    build_logical_blocks(request, BLOCK_SIZE)

    append_decode_token(request, 6, BLOCK_SIZE)

    last = request.logical_blocks[-1]

    assert len(request.logical_blocks) == 2
    assert last.token_ids == [5, 6]
    assert last.block_hash is None


def test_partial_block_becomes_full():
    request = Request(
        request_id="0",
        input_ids=[1, 2, 3, 4, 5, 6, 7],
    )

    build_logical_blocks(request, BLOCK_SIZE)

    append_decode_token(request, 8, BLOCK_SIZE)

    last = request.logical_blocks[-1]

    assert last.token_ids == [5, 6, 7, 8]
    assert last.block_hash is not None

    expected = compute_block_hash(
        request.logical_blocks[0].block_hash,
        [5, 6, 7, 8],
    )

    assert last.block_hash == expected


def test_create_new_block_after_full():
    request = Request(
        request_id="0",
        input_ids=[1, 2, 3, 4],
    )

    build_logical_blocks(request, BLOCK_SIZE)

    append_decode_token(request, 5, BLOCK_SIZE)

    assert len(request.logical_blocks) == 2

    new_block = request.logical_blocks[-1]

    assert new_block.token_ids == [5]
    assert new_block.block_hash is None
    assert new_block.block_idx == 1


def test_multiple_decode_steps():
    request = Request(
        request_id="0",
        input_ids=[1, 2, 3, 4],
    )

    build_logical_blocks(request, BLOCK_SIZE)

    for token in [5, 6, 7, 8, 9]:
        append_decode_token(request, token, BLOCK_SIZE)

    assert len(request.logical_blocks) == 3

    assert request.logical_blocks[1].token_ids == [5, 6, 7, 8]
    assert request.logical_blocks[1].block_hash is not None

    assert request.logical_blocks[2].token_ids == [9]
    assert request.logical_blocks[2].block_hash is None