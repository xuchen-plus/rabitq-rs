//! V4 Manifest + Segment persistence for S3-compatible append-only storage.
//!
//! Layout on disk / object store:
//!
//! ```text
//! {prefix}/
//!   manifest.bin                     ← header + cluster→segment map (small, overwritable)
//!   cluster_0000_v0000.seg           ← immutable segment: one cluster's quantised data
//!   cluster_0001_v0000.seg
//!   ...
//! ```
//!
//! Each segment file is self-contained and never modified.  When vectors are added to a
//! cluster the *entire* cluster is rewritten as a new segment and the manifest is updated.
//! Old segments become dead space (future compaction will clean them up).

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use crc32fast::Hasher;

use crate::{Metric, RabitqError, RotatorType};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const V4_MANIFEST_MAGIC: [u8; 4] = *b"RBQ3";
pub const V4_MANIFEST_VERSION: u32 = 1;
pub const V4_SEGMENT_MAGIC: [u8; 4] = *b"SEG1";
pub const MANIFEST_FILENAME: &str = "manifest.bin";

// ---------------------------------------------------------------------------
// Little-endian helpers
// ---------------------------------------------------------------------------

fn write_u32<W: Write>(w: &mut W, v: u32, h: Option<&mut Hasher>) -> io::Result<()> {
    let b = v.to_le_bytes();
    if let Some(h) = h {
        h.update(&b);
    }
    w.write_all(&b)
}

fn write_u64<W: Write>(w: &mut W, v: u64, h: Option<&mut Hasher>) -> io::Result<()> {
    let b = v.to_le_bytes();
    if let Some(h) = h {
        h.update(&b);
    }
    w.write_all(&b)
}

fn write_f32<W: Write>(w: &mut W, v: f32, h: Option<&mut Hasher>) -> io::Result<()> {
    let b = v.to_le_bytes();
    if let Some(h) = h {
        h.update(&b);
    }
    w.write_all(&b)
}

fn write_u8<W: Write>(w: &mut W, v: u8, h: Option<&mut Hasher>) -> io::Result<()> {
    if let Some(h) = h {
        h.update(&[v]);
    }
    w.write_all(&[v])
}

fn read_u8<R: Read>(r: &mut R, h: Option<&mut Hasher>) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    if let Some(h) = h {
        h.update(&buf);
    }
    Ok(buf[0])
}

fn read_u32<R: Read>(r: &mut R, h: Option<&mut Hasher>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    if let Some(h) = h {
        h.update(&buf);
    }
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R, h: Option<&mut Hasher>) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    if let Some(h) = h {
        h.update(&buf);
    }
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R, h: Option<&mut Hasher>) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    if let Some(h) = h {
        h.update(&buf);
    }
    Ok(f32::from_le_bytes(buf))
}

fn usize_to_u32(v: usize) -> Result<u32, RabitqError> {
    u32::try_from(v)
        .map_err(|_| RabitqError::InvalidPersistence("usize value exceeds u32"))
}

fn usize_to_u64(v: usize) -> Result<u64, RabitqError> {
    u64::try_from(v)
        .map_err(|_| RabitqError::InvalidPersistence("usize value exceeds u64"))
}

fn u32_to_usize(v: u32) -> usize {
    v as usize
}

fn u64_to_usize(v: u64) -> Result<usize, RabitqError> {
    usize::try_from(v)
        .map_err(|_| RabitqError::InvalidPersistence("value exceeds platform usize"))
}

fn metric_to_tag(m: Metric) -> u8 {
    match m {
        Metric::L2 => 0,
        Metric::InnerProduct => 1,
    }
}

fn tag_to_metric(tag: u8) -> Option<Metric> {
    match tag {
        0 => Some(Metric::L2),
        1 => Some(Metric::InnerProduct),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Manifest types
// ---------------------------------------------------------------------------

/// Per-cluster entry in the manifest: which segment file currently holds the data.
#[derive(Debug, Clone)]
pub struct ClusterManifestEntry {
    pub cluster_id: u32,
    pub segment_filename: String,
    pub segment_version: u32,
    pub num_vectors: u32,
    pub file_size: u64,
}

/// Index-level header that is stored in the manifest.
#[derive(Debug, Clone)]
pub struct ManifestHeader {
    pub dim: usize,
    pub padded_dim: usize,
    pub metric: Metric,
    pub rotator_type: RotatorType,
    pub rotator_data: Vec<u8>,
    pub ex_bits: usize,
    pub total_bits: usize,
}

/// Collected segment data for one cluster (in-memory representation).
#[derive(Debug, Clone)]
pub struct ClusterSegmentData {
    pub cluster_id: u32,
    pub centroid: Vec<f32>,
    pub padded_dim: usize,
    pub ex_bits: usize,
    pub ids: Vec<usize>,
    pub batch_data: Vec<u8>,
    pub ex_codes_packed: Vec<Vec<u8>>,
    pub f_add_ex: Vec<f32>,
    pub f_rescale_ex: Vec<f32>,
    pub delta: Vec<f32>,
    pub vl: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Manifest read / write
// ---------------------------------------------------------------------------

/// Load the manifest from a directory path.
pub fn load_manifest(dir: &Path) -> Result<(ManifestHeader, BTreeMap<u32, ClusterManifestEntry>), RabitqError> {
    let manifest_path = dir.join(MANIFEST_FILENAME);
    let file = fs::File::open(&manifest_path)
        .map_err(|e| RabitqError::Io(std::io::Error::new(e.kind(), format!("Cannot open manifest {}: {}", manifest_path.display(), e))))?;
    let mut r = BufReader::new(file);

    // magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != V4_MANIFEST_MAGIC {
        return Err(RabitqError::InvalidPersistence("not a V4 manifest (bad magic)"));
    }

    let version = read_u32(&mut r, None)?;
    if version != V4_MANIFEST_VERSION {
        return Err(RabitqError::InvalidPersistence("unsupported manifest version"));
    }

    let mut hasher = Hasher::new();

    let dim = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);
    let padded_dim = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);
    let metric_tag = read_u8(&mut r, Some(&mut hasher))?;
    let metric = tag_to_metric(metric_tag)
        .ok_or(RabitqError::InvalidPersistence("unknown metric tag in manifest"))?;

    let rotator_type_tag = read_u8(&mut r, Some(&mut hasher))?;
    let rotator_type = RotatorType::from_u8(rotator_type_tag)
        .ok_or(RabitqError::InvalidPersistence("unknown rotator type in manifest"))?;

    let ex_bits = read_u8(&mut r, Some(&mut hasher))? as usize;
    let total_bits = read_u8(&mut r, Some(&mut hasher))? as usize;
    let _total_vectors = read_u64(&mut r, Some(&mut hasher))?;

    let rotator_data_len = u64_to_usize(read_u64(&mut r, Some(&mut hasher))?)?;
    let mut rotator_data = vec![0u8; rotator_data_len];
    r.read_exact(&mut rotator_data)?;
    hasher.update(&rotator_data);

    let cluster_count = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);

    let mut cluster_map: BTreeMap<u32, ClusterManifestEntry> = BTreeMap::new();
    for _ in 0..cluster_count {
        let cluster_id = read_u32(&mut r, Some(&mut hasher))?;
        let segment_version = read_u32(&mut r, Some(&mut hasher))?;
        let num_vectors = read_u32(&mut r, Some(&mut hasher))?;
        let file_size = read_u64(&mut r, Some(&mut hasher))?;

        let fname_len = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);
        let mut fname_bytes = vec![0u8; fname_len];
        r.read_exact(&mut fname_bytes)?;
        hasher.update(&fname_bytes);
        let segment_filename = String::from_utf8(fname_bytes)
            .map_err(|_| RabitqError::InvalidPersistence("non-UTF8 segment filename"))?;

        cluster_map.insert(
            cluster_id,
            ClusterManifestEntry {
                cluster_id,
                segment_filename,
                segment_version,
                num_vectors,
                file_size,
            },
        );
    }

    let computed = hasher.finalize();
    let stored = read_u32(&mut r, None)?;
    if computed != stored {
        return Err(RabitqError::InvalidPersistence("manifest checksum mismatch"));
    }

    let header = ManifestHeader {
        dim: dim as usize,
        padded_dim: padded_dim as usize,
        metric,
        rotator_type,
        rotator_data,
        ex_bits,
        total_bits,
    };

    Ok((header, cluster_map))
}

/// Save the manifest to a directory path.
pub fn save_manifest(
    dir: &Path,
    header: &ManifestHeader,
    cluster_map: &BTreeMap<u32, ClusterManifestEntry>,
) -> Result<(), RabitqError> {
    let manifest_path = dir.join(MANIFEST_FILENAME);
    let tmp_path = dir.join("manifest.tmp");

    {
        let file = fs::File::create(&tmp_path)?;
        let mut w = BufWriter::new(file);
        w.write_all(&V4_MANIFEST_MAGIC)?;
        write_u32(&mut w, V4_MANIFEST_VERSION, None)?;

        let mut hasher = Hasher::new();

        write_u32(&mut w, usize_to_u32(header.dim)?, Some(&mut hasher))?;
        write_u32(&mut w, usize_to_u32(header.padded_dim)?, Some(&mut hasher))?;
        write_u8(&mut w, metric_to_tag(header.metric), Some(&mut hasher))?;
        write_u8(&mut w, header.rotator_type as u8, Some(&mut hasher))?;
        write_u8(&mut w, header.ex_bits as u8, Some(&mut hasher))?;
        write_u8(&mut w, header.total_bits as u8, Some(&mut hasher))?;
        write_u64(&mut w, 0, Some(&mut hasher))?; // total_vectors placeholder (computed from segments)

        write_u64(&mut w, usize_to_u64(header.rotator_data.len())?, Some(&mut hasher))?;
        w.write_all(&header.rotator_data)?;
        hasher.update(&header.rotator_data);

        write_u32(&mut w, usize_to_u32(cluster_map.len())?, Some(&mut hasher))?;

        for entry in cluster_map.values() {
            write_u32(&mut w, entry.cluster_id, Some(&mut hasher))?;
            write_u32(&mut w, entry.segment_version, Some(&mut hasher))?;
            write_u32(&mut w, entry.num_vectors, Some(&mut hasher))?;
            write_u64(&mut w, entry.file_size, Some(&mut hasher))?;
            let fname = entry.segment_filename.as_bytes();
            write_u32(&mut w, usize_to_u32(fname.len())?, Some(&mut hasher))?;
            w.write_all(fname)?;
            hasher.update(fname);
        }

        let checksum = hasher.finalize();
        write_u32(&mut w, checksum, None)?;
        w.flush()?;
    }

    fs::rename(&tmp_path, &manifest_path)?;
    Ok(())
}

/// Build a segment filename from cluster id and version.
pub fn segment_filename(cluster_id: u32, version: u32) -> String {
    format!("cluster_{cluster_id:04}_{version:04}.seg")
}

// ---------------------------------------------------------------------------
// Segment read / write
// ---------------------------------------------------------------------------

/// Read a segment file and return its complete `ClusterSegmentData`.
pub fn read_segment(path: &Path) -> Result<ClusterSegmentData, RabitqError> {
    let file = fs::File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != V4_SEGMENT_MAGIC {
        return Err(RabitqError::InvalidPersistence("not a V4 segment (bad magic)"));
    }

    let mut hasher = Hasher::new();

    let cluster_id = read_u32(&mut r, Some(&mut hasher))?;
    let _segment_version = read_u32(&mut r, Some(&mut hasher))?;
    let padded_dim = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);
    let ex_bits = read_u8(&mut r, Some(&mut hasher))? as usize;
    let num_vectors = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);

    // centroid
    let mut centroid = vec![0.0f32; padded_dim];
    for v in centroid.iter_mut() {
        *v = read_f32(&mut r, Some(&mut hasher))?;
    }

    // ids
    let mut ids = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let v = read_u64(&mut r, Some(&mut hasher))?;
        ids.push(v as usize);
    }

    // batch_data
    let batch_data_len = u64_to_usize(read_u64(&mut r, Some(&mut hasher))?)?;
    let mut batch_data = crate::memory::allocate_aligned_vec::<u8>(batch_data_len);
    r.read_exact(&mut batch_data)?;
    hasher.update(&batch_data);

    // ex_codes_packed
    let ex_code_count = u32_to_usize(read_u32(&mut r, Some(&mut hasher))?);
    let mut ex_codes_packed = Vec::with_capacity(ex_code_count);
    for _ in 0..ex_code_count {
        let len = u64_to_usize(read_u64(&mut r, Some(&mut hasher))?)?;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)?;
        hasher.update(&data);
        ex_codes_packed.push(data);
    }

    // f_add_ex
    let mut f_add_ex = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        f_add_ex.push(read_f32(&mut r, Some(&mut hasher))?);
    }

    // f_rescale_ex
    let mut f_rescale_ex = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        f_rescale_ex.push(read_f32(&mut r, Some(&mut hasher))?);
    }

    // delta
    let mut delta = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        delta.push(read_f32(&mut r, Some(&mut hasher))?);
    }

    // vl
    let mut vl = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        vl.push(read_f32(&mut r, Some(&mut hasher))?);
    }

    let computed = hasher.finalize();
    let stored = read_u32(&mut r, None)?;
    if computed != stored {
        return Err(RabitqError::InvalidPersistence(
            "segment checksum mismatch",
        ));
    }

    Ok(ClusterSegmentData {
        cluster_id,
        centroid,
        padded_dim,
        ex_bits,
        ids,
        batch_data,
        ex_codes_packed,
        f_add_ex,
        f_rescale_ex,
        delta,
        vl,
    })
}

/// Write a `ClusterSegmentData` to a segment file.
pub fn write_segment(path: &Path, seg: &ClusterSegmentData, version: u32) -> Result<u64, RabitqError> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    let mut hasher = Hasher::new();

    w.write_all(&V4_SEGMENT_MAGIC)?;
    write_u32(&mut w, seg.cluster_id, Some(&mut hasher))?;
    write_u32(&mut w, version, Some(&mut hasher))?;
    write_u32(&mut w, usize_to_u32(seg.padded_dim)?, Some(&mut hasher))?;
    write_u8(&mut w, seg.ex_bits as u8, Some(&mut hasher))?;
    write_u32(&mut w, usize_to_u32(seg.ids.len())?, Some(&mut hasher))?;

    // centroid
    for &v in &seg.centroid {
        write_f32(&mut w, v, Some(&mut hasher))?;
    }

    // ids
    for &id in &seg.ids {
        write_u64(&mut w, usize_to_u64(id)?, Some(&mut hasher))?;
    }

    // batch_data
    write_u64(&mut w, usize_to_u64(seg.batch_data.len())?, Some(&mut hasher))?;
    w.write_all(&seg.batch_data)?;
    hasher.update(&seg.batch_data);

    // ex_codes_packed
    write_u32(&mut w, usize_to_u32(seg.ex_codes_packed.len())?, Some(&mut hasher))?;
    for ex in &seg.ex_codes_packed {
        write_u64(&mut w, usize_to_u64(ex.len())?, Some(&mut hasher))?;
        w.write_all(ex)?;
        hasher.update(ex);
    }

    // f_add_ex
    for &v in &seg.f_add_ex {
        write_f32(&mut w, v, Some(&mut hasher))?;
    }
    // f_rescale_ex
    for &v in &seg.f_rescale_ex {
        write_f32(&mut w, v, Some(&mut hasher))?;
    }
    // delta
    for &v in &seg.delta {
        write_f32(&mut w, v, Some(&mut hasher))?;
    }
    // vl
    for &v in &seg.vl {
        write_f32(&mut w, v, Some(&mut hasher))?;
    }

    let checksum = hasher.finalize();
    write_u32(&mut w, checksum, None)?;
    w.flush()?;

    let file_size = w.into_inner()
        .map_err(|_| RabitqError::InvalidPersistence("flush error"))?
        .stream_position()
        .map_err(|e| RabitqError::Io(e))?;

    Ok(file_size)
}

// ---------------------------------------------------------------------------
// Build segment data from a `ClusterData` (existing V3 in-memory format)
// ---------------------------------------------------------------------------

impl ClusterSegmentData {
    /// Build segment data from the in-memory `ClusterData` structure and quantised vectors.
    pub fn from_cluster_data(
        cluster_id: u32,
        centroid: Vec<f32>,
        padded_dim: usize,
        ex_bits: usize,
        ids: Vec<usize>,
        batch_data: Vec<u8>,
        ex_codes_packed: Vec<Vec<u8>>,
        f_add_ex: Vec<f32>,
        f_rescale_ex: Vec<f32>,
        delta: Vec<f32>,
        vl: Vec<f32>,
    ) -> Self {
        Self {
            cluster_id,
            centroid,
            padded_dim,
            ex_bits,
            ids,
            batch_data,
            ex_codes_packed,
            f_add_ex,
            f_rescale_ex,
            delta,
            vl,
        }
    }
}
