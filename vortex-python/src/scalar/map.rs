// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use pyo3::IntoPyObject;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::PyRef;
use pyo3::PyResult;
use pyo3::exceptions::PyIndexError;
use pyo3::pyclass;
use pyo3::pymethods;
use vortex::scalar::MapScalar;
use vortex::scalar::Scalar;

use crate::PyVortex;
use crate::scalar::AsScalarRef;
use crate::scalar::PyScalar;
use crate::scalar::ScalarSubclass;

/// Concrete class for map scalars.
#[pyclass(name = "MapScalar", module = "vortex", extends=PyScalar, frozen)]
pub(crate) struct PyMapScalar;

impl ScalarSubclass for PyMapScalar {
    type Scalar<'a> = MapScalar<'a>;
}

#[pymethods]
impl PyMapScalar {
    /// Return the Python key and value at the given entry index.
    pub fn entry(self_: PyRef<'_, Self>, idx: usize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let scalar = self_.as_scalar_ref();
        let (key, value) = scalar
            .entry(idx)
            .ok_or_else(|| PyIndexError::new_err(format!("Index out of bounds {idx}")))?;
        Ok((
            PyVortex::<&Scalar>(&key).into_pyobject(self_.py())?.into(),
            PyVortex::<&Scalar>(&value)
                .into_pyobject(self_.py())?
                .into(),
        ))
    }
}
