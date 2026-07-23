// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Chunked;
use crate::arrays::Dict;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::slice::SliceExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::mask::MaskExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), Dict, CompareExecuteAdaptor(Dict));
    kernels.register_execute_parent_kernel(Dict.id(), Chunked, TakeExecuteAdaptor(Chunked));
    kernels.register_execute_parent_kernel(Dict.id(), Dict, TakeExecuteAdaptor(Dict));
    kernels.register_execute_parent_kernel(FillNull.id(), Dict, FillNullExecuteAdaptor(Dict));
    kernels.register_execute_parent_kernel(Mask.id(), Dict, MaskExecuteAdaptor(Dict));
    kernels.register_execute_parent_kernel(Slice.id(), Dict, SliceExecuteAdaptor(Dict));
}
