# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest

import vortex as vx


@pytest.mark.parametrize(
    "value,scalar_cls",
    [
        (None, vx.NullScalar),
        (True, vx.BoolScalar),
        (False, vx.BoolScalar),
        (0, vx.PrimitiveScalar),
        (-1, vx.PrimitiveScalar),
        (1.0, vx.PrimitiveScalar),
        ("hello", vx.Utf8Scalar),
        (b"hello", vx.BinaryScalar),
        ({}, vx.StructScalar),
        ({"a": 0, "b": "foo"}, vx.StructScalar),
        ([], vx.ListScalar),
        ([0, 1], vx.ListScalar),
    ],
)
def test_round_trip(
    value: bool | int | float | bytes | str | list[int] | dict[str, str] | None, scalar_cls: type[vx.Scalar]
):
    scalar = vx.scalar(value)
    assert isinstance(scalar, scalar_cls)
    assert scalar.as_py() == value


def test_f16():
    scalar = vx.scalar(1.0, dtype=vx.float_(16))
    assert scalar.dtype == vx.float_(16)
    assert scalar.as_py() == 1.0


def test_map_scalar_from_dict():
    dtype = vx.map_(vx.int_(), vx.utf8(nullable=True), keys_sorted=True, nullable=True)
    scalar = vx.scalar({1: "one", 2: None}, dtype=dtype)

    assert isinstance(scalar, vx.MapScalar)
    assert scalar.dtype == dtype
    assert scalar.as_py() == {1: "one", 2: None}
    key, value = scalar.entry(0)
    assert key == 1
    assert value == "one"


def test_map_scalar_preserves_duplicate_and_unhashable_keys_as_pairs():
    duplicate_dtype = vx.map_(vx.int_(), vx.utf8())
    duplicate = vx.scalar([(1, "first"), (1, "second")], dtype=duplicate_dtype)
    assert isinstance(duplicate, vx.MapScalar)
    assert duplicate.as_py() == [(1, "first"), (1, "second")]

    list_key_dtype = vx.map_(vx.list_(vx.int_()), vx.utf8())
    unhashable = vx.scalar([([1, 2], "value")], dtype=list_key_dtype)
    assert isinstance(unhashable, vx.MapScalar)
    assert unhashable.as_py() == [([1, 2], "value")]
