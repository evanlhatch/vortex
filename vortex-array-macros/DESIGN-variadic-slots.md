# Extending `#[array_slots]` to variable-length slot layouts

## Problem

`#[array_slots]` generates slot index constants, a borrowed view struct, and a typed ext
trait from a struct whose fields are `ArrayRef` or `Option<ArrayRef>`. That only models
encodings with a *fixed* number of slots. Encodings with a variable number of children
cannot use it and hand-roll the same machinery:

| Encoding  | Slot layout                    | Hand-rolled today                        |
| --------- | ------------------------------ | ---------------------------------------- |
| `Chunked` | `[chunk_offsets, chunks...]`   | `CHUNK_OFFSETS_SLOT`, `CHUNKS_OFFSET`    |
| `Struct`  | `[validity?, fields...]`       | `VALIDITY_SLOT`, `FIELDS_OFFSET`         |
| `Union`   | `[type_ids, children...]`      | `TYPE_IDS_SLOT`, `CHILDREN_OFFSET`       |

A key observation about the existing storage model: *optional* slots (`Option<ArrayRef>`)
still occupy a slot index — absence is stored as `None` in place. So optional fields never
shift later indices. The only thing that breaks static indexing is a *variable-length run*
of slots.

## Option A (implemented): single variadic tail — `Vec<ArrayRef>` as the final field

All three variable-length encodings share one shape: a fixed prefix of named slots
followed by exactly one homogeneous variable-length run. Option A models exactly that:
the final field of a slot struct may be `Vec<ArrayRef>`.

```rust
#[array_slots(Union)]
pub struct UnionSlots {
    pub type_ids: ArrayRef,
    pub children: Vec<ArrayRef>,
}
```

Generated API:

```rust
impl UnionSlots {
    pub const TYPE_IDS: usize = 0;
    /// Offset at which the `children` slots begin.
    pub const CHILDREN_OFFSET: usize = 1;
    /// Number of fixed (non-variadic) slots.
    pub const FIXED_COUNT: usize = 1;
    pub const FIXED_NAMES: [&'static str; 1] = ["type_ids"];
    /// "type_ids" for 0, "children[i]" for the tail.
    pub fn slot_name(idx: usize) -> String;

    pub fn from_slots(slots: ArraySlots) -> Self;   // drains the tail
    pub fn into_slots(self) -> ArraySlots;          // appends the tail
}

pub struct UnionSlotsView<'a> {
    pub type_ids: &'a ArrayRef,
    pub children: SlotSlice<'a>,   // Copy; len()/get()/iter()/Index
}

pub trait UnionArraySlotsExt: TypedArrayRef<Union> {
    fn type_ids(&self) -> &ArrayRef;
    fn children(&self) -> SlotSlice<'_>;
    fn slots_view(&self) -> UnionSlotsView<'_>;
}
```

`SlotSlice<'a>` is a new core type in `vortex-array`: a borrowed run of *required* slots
(`&[Option<ArrayRef>]` whose entries validation guarantees are present), exposing
`len()`, `get()`, `iter()`, `to_vec()`, and `Index<usize>` without per-call-site
`.vortex_expect(...)`.

`Union` has been migrated to this in the same change as a proof of fit: the hand-rolled
constants are gone, and the hand-written `UnionArrayExt` now layers only *domain* lookups
(`variants()`, `child_by_type_id()`, `child_by_name()`) on top of the generated
`UnionArraySlotsExt` supertrait. `Chunked` and `Struct` migrate the same way (their
cached derived data — e.g. `ChunkedData::chunk_offsets` — is orthogonal to the slot
layout and unaffected).

Non-goals / limits of Option A:

- At most one variadic run, and it must be trailing. This is not incidental: with a single
  trailing run, every fixed slot keeps a compile-time index and the run's extent is
  `FIXED_COUNT..len`. Nothing about the array besides `slots.len()` is needed to interpret
  the layout.
- The tail is a run of *required* slots. A `Vec<Option<ArrayRef>>` tail (optional entries
  inside the run) is expressible in storage but has no current user; it can be added later
  without breaking anything.

## Option B (sketched): multiple counted sections

If an encoding ever needs *two* variable-length runs — e.g. a chunked union that stores
per-chunk type IDs *and* per-chunk children — slot count alone can no longer locate the
boundary. The split must come from somewhere the array already knows, which in practice
means the `DType` (or the encoding's `TypedArrayData`).

Sketch:

```rust
#[array_slots(ChunkedUnion)]
pub struct ChunkedUnionSlots {
    pub chunk_offsets: ArrayRef,
    #[slots(count = |dtype: &DType| dtype.as_union_variants().len())]
    pub variant_metas: Vec<ArrayRef>,   // counted section: length from dtype
    pub chunks: Vec<ArrayRef>,          // final section: takes the remainder
}
```

Generated conversions become dtype-aware, and offsets become runtime values:

```rust
impl ChunkedUnionSlots {
    pub const CHUNK_OFFSETS: usize = 0;
    pub const VARIANT_METAS_OFFSET: usize = 1;
    pub fn chunks_offset(dtype: &DType) -> usize;              // 1 + count(dtype)
    pub fn from_slots(slots: ArraySlots, dtype: &DType) -> Self;
}

pub trait ChunkedUnionArraySlotsExt: TypedArrayRef<ChunkedUnion> {
    fn variant_metas(&self) -> SlotSlice<'_>;   // splits using self.dtype() internally
    fn chunks(&self) -> SlotSlice<'_>;
}
```

The ext trait stays ergonomic (it has `self`, hence the dtype), but the plain
`from_slots`/view constructors grow a `&DType` parameter for every encoding that uses a
counted section. Deferred because no in-tree encoding needs it yet — the chunked-union
canonicalization TODO is the first candidate, and its representation is still undecided.

## Option C (sketched): grouped tail — `Vec<Group>` of repeating slot tuples

Some layouts repeat a *tuple* of slots per element rather than a single slot — e.g. a
per-chunk patch pair `[indices, values]`. Flattened storage would be
`[fixed..., i0, v0, i1, v1, ...]`. Sketch: a lightweight `#[slot_group]` derive for the
repeating unit, referenced as the tail element type:

```rust
#[slot_group]
pub struct PatchPair {
    pub indices: ArrayRef,
    pub values: ArrayRef,
}

#[array_slots(PatchedChunks)]
pub struct PatchedChunksSlots {
    pub inner: ArrayRef,
    pub patches: Vec<PatchPair>,   // group count = (slots.len() - FIXED_COUNT) / PatchPair::COUNT
}
```

`#[slot_group]` generates `PatchPair::COUNT`, `PatchPairView<'a>`, and per-group
conversions; `#[array_slots]` chunks the tail by `PatchPair::COUNT`. The ext accessor
returns a `GroupSlice<'a, PatchPair>` whose `get(i)` yields a `PatchPairView<'a>`, which
requires a linking trait:

```rust
pub trait SlotGroup {
    const COUNT: usize;
    type View<'a>;
    fn view(slots: &[Option<ArrayRef>]) -> Self::View<'_>;
}
```

Deferred: no in-tree encoding stores repeating tuples today (`ALP-RD` and `Patched` keep
patches in a fixed number of slots by concatenating across chunks and storing lane/chunk
offsets — arguably the better representation anyway, since it keeps slot count O(1)).

## Option D (rejected): runtime slot-schema object

Replace the macro with a runtime description (`SlotSchema::new().slot("type_ids").tail("children")`)
attached to the vtable. Rejected: loses compile-time constants, typed views, and per-field
accessor inlining; every access goes through string/enum lookup; and the macro's main
value — the struct doubling as a constructor-side transport type — disappears.

## Trade-offs

| | A: variadic tail | B: counted sections | C: grouped tail | D: runtime schema |
| --- | --- | --- | --- | --- |
| Covers `Chunked`/`Struct`/`Union` | yes | yes | yes | yes |
| Covers multi-run layouts | no | yes | tuples only | yes |
| Compile-time offsets | fixed prefix | prefix only, rest dtype-dependent | fixed prefix | none |
| `from_slots` signature | unchanged | gains `&DType` | unchanged | n/a |
| Macro complexity | small delta | medium (count expressions, runtime offsets) | medium (extra derive + linking trait) | replaces macro |
| In-tree demand today | 3 encodings | 0 | 0 | — |

Option A is implemented because it covers every current variable-length encoding at
near-zero complexity cost and keeps the fixed-slot behavior byte-for-byte identical.
B and C are forward-compatible extensions of the same surface: a counted section or a
group tail can be added later without changing anything Option A generated, because both
reuse the "fixed prefix + tail" storage discipline and only refine how the tail is
interpreted.
