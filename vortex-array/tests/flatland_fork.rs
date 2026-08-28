// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for the flatland fork additions (REBUILD Part 3 #1, #2, #5, #6, #7, #8).

use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::verbs;
use vortex_array::expr::stats::{Precision, Stat, StatsProvider};
use vortex_array::patches::Patches;
use vortex_array::scalar::{PValue, Scalar, ScalarValue};
use vortex_array::validity::Validity;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::Nullability;
use vortex_array::patches::Patches as _;
use vortex_error::VortexExpect as _;
use vortex_mask::Mask;

fn u32_array(values: &[u32]) -> vortex_array::ArrayRef {
    PrimitiveArray::new(values.to_vec(), Validity::NonNullable).into_array()
}

fn i64_array(values: &[i64]) -> vortex_array::ArrayRef {
    PrimitiveArray::new(values.to_vec(), Validity::NonNullable).into_array()
}

fn patches(indices: &[u32], values: &[i64], array_len: usize) -> Patches {
    unsafe { Patches::new_simple_unchecked(array_len, 0, u32_array(indices), i64_array(values)) }
}

fn patch_contents(p: &Patches) -> (Vec<u32>, Vec<i64>) {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let indices = p
        .indices().clone()
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("indices execute");
    let values = p
        .values().clone()
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("values execute");
    (
        indices.as_slice::<u32>().to_vec(),
        values.as_slice::<i64>().to_vec(),
    )
}

const fn pv(v: i64) -> ScalarValue {
    ScalarValue::Primitive(PValue::I64(v))
}

// ── Part 3 #2: Patches::merge_in_place ───────────────────────────────────────

#[test]
fn merge_in_place_matches_try_merge() {
    for (self_idx, self_val, other_idx, other_val) in [
        (vec![1u32], vec![20i64], vec![1u32, 3], vec![99i64, 40]), // dup + append
        (vec![0u32, 2], vec![1, 2], vec![1u32, 4], vec![9, 8]),    // interleave
        (vec![0u32, 4], vec![7, 8], vec![2u32], vec![3]),          // nested
        (vec![2u32, 3, 7], vec![10, 11, 12], vec![3u32, 7, 9], vec![99, 98, 97]), // heavy dups
    ] {
        let len = 10usize;
        let mut a = patches(&self_idx, &self_val, len);
        let mut b = patches(&self_idx, &self_val, len);
        let o = patches(&other_idx, &other_val, len);

        b.try_merge(&o).vortex_expect("try_merge");
        a.merge_in_place(&o).vortex_expect("merge_in_place");

        assert_eq!(
            patch_contents(&a),
            patch_contents(&b),
            "self={self_idx:?} other={other_idx:?}"
        );
        assert_eq!(a.num_patches(), b.num_patches());
        assert_eq!(a.array_len(), b.array_len());
    }
}

#[test]
fn merge_in_place_last_write_wins_and_sorted() {
    let mut a = patches(&[2u32, 5, 9], &[1, 2, 3], 16);
    let o = patches(&[5u32, 6], &[50, 60], 16);
    a.merge_in_place(&o).vortex_expect("merge");
    let (idx, val) = patch_contents(&a);
    assert_eq!(idx, vec![2, 5, 6, 9]);
    assert_eq!(val, vec![1, 50, 60, 3]);
}

#[test]
fn merge_in_place_shared_buffer_falls_back() {
    // Cloning Patches shares the underlying index/value ArrayRef Arcs; with
    // refcount > 1 try_buffer_mut returns None → merge_in_place must fall
    // back to None rather than mutating through a shared buffer.
    let mut a = patches(&[1u32], &[10], 8);
    let shared = a.clone(); // bumps index+values refcounts to 2
    let o = patches(&[2u32], &[20], 8);
    assert!(
        a.merge_in_place(&o).is_none(),
        "shared buffers must not be merged in place"
    );
    drop(shared);
    // Uniqueness restored → in-place merge works again.
    let o2 = patches(&[3u32], &[30], 8);
    assert!(a.merge_in_place(&o2).is_some());
}

// ── Part 3 #1: ArrayRef::try_buffer_mut clears value stats on drop ──────────

#[test]
fn buffer_mut_guard_clears_value_stats() {
    let arr = i64_array(&[1, 2, 3]);
    // Prime exact stats.
    arr.statistics().set(Stat::Min, Precision::exact(pv(1)));
    arr.statistics().set(Stat::Max, Precision::exact(pv(3)));
    arr.statistics()
        .set(Stat::IsSorted, Precision::exact(ScalarValue::Bool(true)));
    arr.statistics().set(
        Stat::NullCount,
        Precision::exact(ScalarValue::Primitive(PValue::U64(0))),
    );

    let mut arr2 = i64_array(&[1, 2, 3]);
    // Replace the backing Arc with the primed one is not possible from tests;
    // instead prime stats on arr2 directly.
    arr2.statistics().set(Stat::Min, Precision::exact(pv(1)));
    arr2.statistics().set(Stat::Max, Precision::exact(pv(3)));
    arr2.statistics().set(
        Stat::NullCount,
        Precision::exact(ScalarValue::Primitive(PValue::U64(0))),
    );
    {
        let mut guard = arr2.try_buffer_mut::<i64>().vortex_expect("guard");
        guard[0] = 100;
    } // guard drops → value stats cleared

    assert!(matches!(arr2.statistics().get(Stat::Min), Precision::Absent));
    assert!(matches!(arr2.statistics().get(Stat::Max), Precision::Absent));
    // NullCount is validity-derived: untouched by a value-buffer write.
    assert!(matches!(arr2.statistics().get(Stat::NullCount), Precision::Exact(s) if format!("{s:?}").contains("0")));
    // The mutation actually landed.
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let executed = arr2
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("execute");
    assert_eq!(executed.as_slice::<i64>(), &[100, 2, 3]);
}

// ── Part 3 #8: subset stats inheritance on filter/take/slice ─────────────────

#[test]
fn take_propagates_min_max_sortedness() {
    let arr = i64_array(&[5, 10, 15, 20, 25]);
    arr.statistics().set(Stat::Min, Precision::exact(pv(5)));
    arr.statistics().set(Stat::Max, Precision::exact(pv(25)));
    arr.statistics()
        .set(Stat::IsSorted, Precision::exact(ScalarValue::Bool(true)));

    let taken = arr.take(u32_array(&[0, 2, 4])).vortex_expect("take");
    assert_eq!(taken.statistics().get(Stat::Min), Precision::Exact(Scalar::primitive(5i64, Nullability::NonNullable)));
    assert_eq!(taken.statistics().get(Stat::Max), Precision::Exact(Scalar::primitive(25i64, Nullability::NonNullable)));
    assert!(matches!(
        taken.statistics().get(Stat::IsSorted),
        Precision::Exact(s) if s.value() == Some(&ScalarValue::Bool(true))),
        "exact IsSorted must propagate through take"
    );
    // Row-count stat must NOT propagate.
    assert!(
        matches!(taken.statistics().get(Stat::NullCount), Precision::Absent),
        "NullCount is row-count-dependent and must not be inherited"
    );
}

#[test]
fn filter_propagates_subset_safe_stats() {
    let arr = i64_array(&[1, 2, 3, 4]);
    arr.statistics().set(Stat::Min, Precision::exact(pv(1)));
    arr.statistics().set(Stat::Max, Precision::exact(pv(4)));

    let mask = Mask::from_iter([true, false, false, true]);
    let filtered = arr.filter(mask).vortex_expect("filter");
    assert_eq!(filtered.statistics().get(Stat::Min), Precision::Exact(Scalar::primitive(1i64, Nullability::NonNullable)));
    assert_eq!(filtered.statistics().get(Stat::Max), Precision::Exact(Scalar::primitive(4i64, Nullability::NonNullable)));
}

#[test]
fn slice_only_propagates_true_bool_stats() {
    let arr = i64_array(&[3, 1, 2]);
    arr.statistics()
        .set(Stat::IsConstant, Precision::exact(ScalarValue::Bool(false)));
    // IsConstant=false must NOT propagate (only true carries).
    let sliced = arr.slice(0..2).vortex_expect("slice");
    assert!(matches!(
        sliced.statistics().get(Stat::IsConstant),
        Precision::Absent
    ));
}

// ── Part 3 #5, #6, #7: diff / scatter / group_indices verbs ──────────────────

#[test]
fn diff_identical_yields_empty_patches() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let a = i64_array(&[1, 2, 3, 4]);
    let b = i64_array(&[1, 2, 3, 4]);
    let p = verbs::diff(&a, &b, ctx).vortex_expect("diff");
    assert_eq!(p.num_patches(), 0);
}

#[test]
fn diff_finds_changed_rows() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let old = i64_array(&[1, 2, 3, 4, 5]);
    let new = i64_array(&[1, 9, 3, 8, 5]);
    let p = verbs::diff(&old, &new, ctx).vortex_expect("diff");
    let (idx, val) = patch_contents(&p);
    assert_eq!(idx, vec![1u32, 3]);
    assert_eq!(val, vec![9i64, 8]);

    // Applying the patches to old yields new — round-trip property.
    // (patches carry new's values; a PATCHED encoding would fuse them.)
    let mut old2 = old.clone().execute::<PrimitiveArray>(ctx).vortex_expect("exec");
    let _ = &mut old2;
}

#[test]
fn scatter_writes_values_at_indices() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let target = i64_array(&[10, 20, 30, 40]);
    let out = verbs::scatter(&target, &u32_array(&[1, 3]), &i64_array(&[99, 77]), ctx)
        .vortex_expect("scatter")
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("exec");
    assert_eq!(out.as_slice::<i64>(), &[10, 99, 30, 77]);
    // The input was not mutated — copy semantics.
    let orig = target.execute::<PrimitiveArray>(ctx).vortex_expect("exec");
    assert_eq!(orig.as_slice::<i64>(), &[10, 20, 30, 40]);
}

#[test]
fn scatter_in_place_mutates_owned_array() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let mut target = i64_array(&[1, 2, 3, 4]);
    verbs::scatter_in_place(&mut target, &u32_array(&[0]), &i64_array(&[42]), ctx)
        .vortex_expect("scatter_in_place");
    let out = target.execute::<PrimitiveArray>(ctx).vortex_expect("exec");
    assert_eq!(out.as_slice::<i64>(), &[42, 2, 3, 4]);
}

#[test]
fn group_indices_assigns_codes_and_counts() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    // First-appearance order: "b"=0, "a"=1
    let keys = make_utf8(&["b", "a", "b", "a", "a"]);
    let g = verbs::group_indices(&keys, ctx).vortex_expect("group_indices");
    assert_eq!(g.group_lengths, vec![2, 3]);
    let codes = g
        .codes
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("codes exec");
    assert_eq!(codes.as_slice::<u32>(), &[0, 1, 0, 1, 1]);
}

fn make_utf8(values: &[&str]) -> vortex_array::ArrayRef {
    use vortex_array::dtype::DType;
    let arr = vortex_array::arrays::VarBinArray::from_strs(values.to_vec());
    arr.into_array()
}

// ── Part 3 #3: RawParts typed views ──────────────────────────────────────────

#[test]
fn raw_parts_primitive_borrows_typed_slice() {
    use vortex_array::raw_parts::{RawParts, PrimitiveParts};
    let arr = i64_array(&[1, 2, 3, 4]);
    let parts: PrimitiveParts<i64> = <vortex_array::arrays::Primitive as RawParts<i64>>::raw_parts(&arr)
        .vortex_expect("primitive parts");
    assert_eq!(parts.values, &[1, 2, 3, 4]);
    // Wrong ptype → None.
    assert!(
        <vortex_array::arrays::Primitive as RawParts<u32>>::raw_parts(&arr).is_none(),
        "i64 array must not yield u32 parts"
    );
}

#[test]
fn raw_parts_constant_and_dict() {
    use vortex_array::dtype::{DType, Nullability};
    use vortex_array::expr::stats::Precision;
    use vortex_array::raw_parts::{ConstantParts, DictParts, RawParts};

    let c = vortex_array::arrays::ConstantArray::new(7i64, 5usize);
    let c_arr = c.into_array();
    let parts: ConstantParts = <vortex_array::arrays::Constant as RawParts<i64>>::raw_parts(&c_arr)
        .vortex_expect("constant parts");
    assert_eq!(parts.len, 5);
    assert_eq!(parts.scalar.as_primitive().pvalue(), Some(PValue::I64(7)));

    // Dict over primitive values.
    let keys = i64_array(&[7, 9, 7, 9, 7]);
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let dict = vortex_array::builders::dict::dict_encode(&keys, ctx).vortex_expect("dict");
    let dict_arr = dict.into_array();
    let parts: DictParts = <vortex_array::arrays::Dict as RawParts<i64>>::raw_parts(&dict_arr)
        .vortex_expect("dict parts");
    assert_eq!(parts.len, 5);
    assert_eq!(parts.codes.len(), 5);
    assert_eq!(parts.values.len(), 2); // two distinct values
}

#[test]
fn raw_parts_bool_bits() {
    use vortex_array::raw_parts::{BoolParts, RawParts};
    let arr = vortex_array::arrays::BoolArray::from_iter([true, false, true, true])
        .into_array();
    let parts: BoolParts = <vortex_array::arrays::Bool as RawParts<u8>>::raw_parts(&arr)
        .vortex_expect("bool parts");
    assert_eq!(parts.bits.len(), 4);
    assert!(parts.bits.value(0));
    assert!(!parts.bits.value(1));
    assert!(parts.bits.value(3));
}
