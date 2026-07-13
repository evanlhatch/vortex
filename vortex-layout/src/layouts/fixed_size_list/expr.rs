// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_error::VortexResult;

/// The minimal set of fixed-size-list children an expression needs for evaluation.
///
/// For example:
///     - `is_null(root())` only needs validity.
///     - `list_length(root())` needs the fixed list size and validity, but not elements.
///     - `root()` needs elements and validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum FixedSizeListChildrenNeeded {
    /// Only validity is needed (`is_null` / `is_not_null`).
    Validity,
    /// Only the fixed list size and validity are needed (`list_length`).
    ListLengthAndValidity,
    /// Elements and validity are needed.
    Elements,
}

/// The minimal set of fixed-size-list children needed to evaluate `expr`, where `root()` is a
/// field with fixed-size-list dtype.
pub(super) fn get_necessary_fixed_size_list_children(
    expr: &Expression,
) -> FixedSizeListChildrenNeeded {
    if is_null_root(expr) {
        return FixedSizeListChildrenNeeded::Validity;
    }

    if is_list_length_root(expr) {
        return FixedSizeListChildrenNeeded::ListLengthAndValidity;
    }

    if is_root(expr) {
        return FixedSizeListChildrenNeeded::Elements;
    }

    expr.children()
        .iter()
        .map(get_necessary_fixed_size_list_children)
        .max()
        .unwrap_or(FixedSizeListChildrenNeeded::Validity)
}

fn is_null_root(expr: &Expression) -> bool {
    (expr.is::<IsNull>() || expr.is::<IsNotNull>())
        && expr.children().len() == 1
        && is_root(expr.child(0))
}

fn is_list_length_root(expr: &Expression) -> bool {
    expr.is::<ListLength>() && expr.children().len() == 1 && is_root(expr.child(0))
}

/// Rewrite a validity-class expression so it can be evaluated against the fixed-size-list's
/// validity bool array (`true` == valid row): `is_not_null(root())` becomes `root()` and
/// `is_null(root())` becomes `not(root())`. All other nodes are rebuilt with rewritten children.
pub(super) fn rewrite_validity_expr(expr: &Expression) -> VortexResult<Expression> {
    if expr.is::<IsNotNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(root());
    }
    if expr.is::<IsNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(not(root()));
    }
    let children = expr
        .children()
        .iter()
        .map(rewrite_validity_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// Rewrite a list-length-class expression so it can be evaluated against an array of list lengths.
/// `list_length(root())` becomes `root()`. Other references to `root()` are left intact: for
/// list-length-class expressions they can only be validity checks, and the lengths array carries
/// the same validity as the original fixed-size-list.
pub(super) fn rewrite_list_length_expr(expr: &Expression) -> VortexResult<Expression> {
    if is_list_length_root(expr) {
        return Ok(root());
    }

    let children = expr
        .children()
        .iter()
        .map(rewrite_list_length_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}
