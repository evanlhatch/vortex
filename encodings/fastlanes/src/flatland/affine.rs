// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Affine transform in the encoded domain (flatland REBUILD Part 0:
//! "absorb's 3 sound arms" — the ECS math primitive `dst = v*factor + base`).
//!
//! Arithmetic-on-encoded does NOT exist upstream at 0.85 (for/bitpacking/alp
//! compute dirs carry compare/cast/filter/take only). This kernel supplies
//! the affine case without canonicalizing, per arm:
//!
//! 1. **Constant** → O(1) scalar rewrite, new Constant broadcast.
//! 2. **FoR, factor==1** → **ref-bump**: rebuild FoR with `reference + base`,
//!    encoded child untouched — width-preserving, zero row work. Overflow of
//!    the ptype range is an error (out-of-range constant).
//! 3. **Dict** → **values-map**: recurse on the dict `values` array
//!    (O(|dict|) distinct rows, not O(n)); codes are reused verbatim.
//! 4. **Primitive** → elementwise wrap-mul/wrap-add over the buffer (u32
//!    add-only rides the portable SIMD add-const tier).
//!
//! FoR with `factor != 1` decompresses to the Primitive path: scaling widens
//! bit-width, so a facts-gated re-pack is a re-encode policy (REBUILD
//! Part T2.2), not this kernel. Float/decimal dtypes are rejected — the
//! flatland convention is u*-only keys, i* via FoR, f* via ALP-as-terminal.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictSlots;
use vortex_array::arrays::primitive::PrimitiveArrayExt as _;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_ensure;

use crate::FoR;
use crate::FoRArrayExt as _;
use crate::FoRSlots;
use crate::r#for::decompress;

/// Apply `dst = v*factor + base` in the encoded domain where sound.
///
/// Integer dtypes only (flatland convention: u*-only keys, i* via FoR, f* via
/// ALP — floats never ride this kernel). Scalar arms check overflow of the
/// logical dtype; the Primitive row loop is modular (wrapping) arithmetic —
/// logical range validation belongs to the compiler's constant/gate checks,
/// not the hot loop.
pub fn affine(
    column: &ArrayRef,
    factor: i64,
    base: i64,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // 1. Constant: O(1) scalar rewrite.
    if let Some(c) = column.as_opt::<Constant>() {
        let scalar = c.scalar();
        vortex_ensure!(
            !scalar.is_null(),
            "affine on null constant is not representable"
        );
        let ptype = scalar.dtype().as_ptype();
        let nullability = scalar.dtype().nullability();
        let pv = scalar
            .as_primitive()
            .pvalue()
            .ok_or_else(|| vortex_err!("constant is not a primitive value"))?;
        let new = affine_pvalue(pv, factor, base)?;
        let out = Scalar::primitive_value(new, ptype, nullability);
        return Ok(ConstantArray::new(out, column.len()).into_array());
    }

    // 2. FoR, add-only: ref-bump, encoded child untouched.
    if column.is::<FoR>() {
        if factor == 1 {
            let f = column.as_::<FoR>();
            let ref_scalar = f.reference_scalar();
            let ref_ptype = ref_scalar.dtype().as_ptype();
            let ref_null = ref_scalar.dtype().nullability();
            let pv = ref_scalar
                .as_primitive()
                .pvalue()
                .ok_or_else(|| vortex_err!("FoR reference is not a primitive value"))?;
            let bumped = affine_pvalue(pv, 1, base)?;
            let encoded = f
                .as_ref()
                .slots()
                .get(FoRSlots::ENCODED)
                .and_then(|s| s.as_ref())
                .ok_or_else(|| vortex_err!("FoR array missing ENCODED slot"))?
                .clone();
            // try_new re-validates the reference dtype; the encoded child
            // moves by refcount — zero row work.
            let rebuilt = FoR::try_new(
                encoded,
                Scalar::primitive_value(bumped, ref_ptype, ref_null),
            )?;
            return Ok(rebuilt.into_array());
        }
        // factor != 1: decompress to the primitive path (scale widens
        // bit-width; facts-gated re-pack is Part T2.2, not this kernel).
        let owned = column.clone().downcast::<FoR>();
        let p = decompress(&owned, ctx)?;
        return affine(&p.into_array(), factor, base, ctx);
    }

    // 3. Dict: map the distinct values, reuse codes verbatim (O(|dict|)).
    if let Some(d) = column.as_opt::<Dict>() {
        let codes = d
            .as_ref()
            .slots()
            .get(DictSlots::CODES)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| vortex_err!("Dict array missing CODES slot"))?
            .clone();
        let values = d
            .as_ref()
            .slots()
            .get(DictSlots::VALUES)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| vortex_err!("Dict array missing VALUES slot"))?
            .clone();
        let new_values = affine(&values, factor, base, ctx)?;
        return Ok(DictArray::try_new(codes, new_values)?.into_array());
    }

    // 4. Primitive: elementwise wrap-mul/wrap-add.
    if let Some(p) = column.as_opt::<Primitive>() {
        let ptype = p.ptype();
        if !ptype.is_int() {
            return Err(vortex_err!(
                "affine supports integer dtypes only, got {:?} (flatland: u* keys, i* via FoR, f* via ALP)",
                ptype
            ));
        }
        let validity = p.validity()?;

        // u32 add-only rides the portable SIMD add-const tier.
        if ptype == PType::U32 && factor == 1 {
            let src = p.as_slice::<u32>();
            let mut out = vec![0u32; src.len()];
            vortex_buffer::portable::add_const_u32(src, base as u32, &mut out);
            return Ok(PrimitiveArray::new(out, validity).into_array());
        }

        let out = match_each_integer_ptype!(ptype, |T| {
            // `as T` truncation is sound here: affine is modular in the
            // element type — (a*b) mod 2^N == (a*(b mod 2^N)) mod 2^N.
            let f = factor as T;
            let b = base as T;
            let out: Vec<T> = p
                .as_slice::<T>()
                .iter()
                .map(|&v| v.wrapping_mul(f).wrapping_add(b))
                .collect();
            PrimitiveArray::new(out, validity)
        });
        return Ok(out.into_array());
    }

    Err(vortex_err!(
        "affine: unsupported encoding {:?} (supported: Constant, FoR, Dict, Primitive)",
        column.encoding_id(),
    ))
}

/// Scalar rewrite `v*factor + base` over a `PValue`, preserving dtype width.
/// Arithmetic in i64 with checked narrowing — overflow of the ptype range is
/// an error, never a silent wrap.
fn affine_pvalue(pv: PValue, factor: i64, base: i64) -> VortexResult<PValue> {
    let v: i64 = pvalue_to_i64(pv)?;
    let r = v
        .checked_mul(factor)
        .and_then(|x| x.checked_add(base))
        .ok_or_else(|| vortex_err!("affine: {}*{}+{} overflows i64", v, factor, base))?;
    match pv {
        PValue::U8(_) => Ok(PValue::U8(u8::try_from(r)?)),
        PValue::U16(_) => Ok(PValue::U16(u16::try_from(r)?)),
        PValue::U32(_) => Ok(PValue::U32(u32::try_from(r)?)),
        PValue::U64(_) => Ok(PValue::U64(u64::try_from(r)?)),
        PValue::I8(_) => Ok(PValue::I8(i8::try_from(r)?)),
        PValue::I16(_) => Ok(PValue::I16(i16::try_from(r)?)),
        PValue::I32(_) => Ok(PValue::I32(i32::try_from(r)?)),
        PValue::I64(_) => Ok(PValue::I64(r)),
        other => Err(vortex_err!(
            "affine: non-integer PValue {:?} not supported",
            other
        )),
    }
}

fn pvalue_to_i64(pv: PValue) -> VortexResult<i64> {
    Ok(match pv {
        PValue::U8(v) => i64::from(v),
        PValue::U16(v) => i64::from(v),
        PValue::U32(v) => i64::from(v),
        PValue::U64(v) => i64::try_from(v)?,
        PValue::I8(v) => i64::from(v),
        PValue::I16(v) => i64::from(v),
        PValue::I32(v) => i64::from(v),
        PValue::I64(v) => v,
        other => return Err(vortex_err!("affine: non-integer PValue {:?}", other)),
    })
}
