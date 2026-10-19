"""A tag write that cannot be stored must raise, never print and return.

``OQPData.__setitem__`` used to wrap the dtype lookup and the native
``oqp_alloc`` call in a bare ``except:`` that printed "The type of your data
is not allowed" and returned.  Every failed tag write was then silent, so a
caller such as the SCF stability safeguard's state restore continued with the
old contents of the tag.  These tests require that an unsupported dtype raises
TypeError naming the tag and dtype, and that a failing ``oqp_alloc``
propagates as an exception, while supported writes still round-trip.
"""
import numpy as np
import pytest


@pytest.fixture
def data():
    from oqp.molecule.oqpdata import OQPData
    return OQPData()


@pytest.mark.parametrize("value", [
    np.array([True, False]),
    np.array(["a", "bc"]),
    np.array([1.0, 2.0], dtype=np.float16),
    np.array([None, 1], dtype=object),
])
def test_unsupported_dtype_raises_type_error(data, value):
    key = "OQP::PROBE_UNSUPPORTED_DTYPE"
    with pytest.raises(TypeError) as err:
        data[key] = value
    message = str(err.value)
    assert key in message
    assert value.dtype.str in message


def test_supported_dtypes_round_trip(data):
    # Non-symmetric values so a transposed or reordered copy is visible.
    f64 = np.arange(6, dtype=np.float64).reshape(2, 3) * 1.5 - 2.0
    i64 = np.array([3, -1, 7], dtype=np.int64)
    i32 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    data["OQP::PROBE_F64"] = f64
    data["OQP::PROBE_I64"] = i64
    data["OQP::PROBE_I32"] = i32
    data["OQP::PROBE_STR"] = "probe"
    np.testing.assert_array_equal(np.asarray(data["OQP::PROBE_F64"]), f64)
    np.testing.assert_array_equal(np.asarray(data["OQP::PROBE_I64"]), i64)
    np.testing.assert_array_equal(np.asarray(data["OQP::PROBE_I32"]), i32)
    assert data["OQP::PROBE_STR"] == "probe"


class _FailingAlloc:
    """Delegate to the real cffi lib but make ``oqp_alloc`` report failure."""

    def __init__(self, real, status):
        self._real = real
        self._status = status

    def oqp_alloc(self, *args):
        return self._status

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.parametrize("status, exc", [(-2, MemoryError), (-1, RuntimeError)])
def test_failed_native_alloc_propagates(data, monkeypatch, status, exc):
    import oqp.molecule.oqpdata as oqpdata_mod

    key = "OQP::PROBE_ALLOC_FAILURE"
    data[key] = np.array([1.0, 2.0])
    monkeypatch.setattr(oqpdata_mod, "lib", _FailingAlloc(oqpdata_mod.lib, status))
    with pytest.raises(exc) as err:
        data[key] = np.array([5.0, 6.0, 7.0])
    assert key in str(err.value)
    monkeypatch.undo()
    # The failed write left the previous contents in place; the caller saw
    # an exception instead of silently continuing with them.
    np.testing.assert_array_equal(np.asarray(data[key]), [1.0, 2.0])


def _tag_status(data, key):
    """Raw ``oqp_get`` return for ``key``: >= 0 when a tag record exists."""
    from oqp import ffi, lib
    from oqp.molecule.oqpdata import TA_DIMENSIONS_LENGTH

    return lib.oqp_get(data._data, ffi.new('char []', bytes(key, 'ascii')),
                       ffi.new('int32_t *'), ffi.new('int32_t *'),
                       ffi.new(f'int64_t[{TA_DIMENSIONS_LENGTH}]'),
                       ffi.new('void **'))


@pytest.mark.parametrize("key, value, structs", [
    ("tda", True, ("tddft",)),
    ("debug_mode", True, ("tddft", "mpiinfo")),
    ("usempi", False, ("mpiinfo",)),
    ("charge", -1, ("mol_prop",)),
])
def test_struct_member_write_is_not_stored_as_tag(data, key, value, structs):
    # A struct member (a bool here, which tagarray cannot store) is written to
    # every native struct that has it and the call returns; it must neither
    # raise after the member changed nor leave a stray tag of the same name.
    data[key] = value
    for name in structs:
        assert getattr(getattr(data._data, name), key) == value
    assert data[key] == value
    assert _tag_status(data, key) < 0


def test_refused_native_create_raises_before_copy(data):
    # tagarray rejects a tag containing a control character.  oqp_alloc must
    # report that refusal itself (-3) rather than look the tag up afterwards.
    key = "OQP::PROBE\x01REFUSED"
    with pytest.raises(ValueError) as err:
        data[key] = np.array([1.0, 2.0, 3.0])
    assert "refused" in str(err.value)
    assert _tag_status(data, key) < 0
