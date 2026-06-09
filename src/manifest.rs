//! V4 Manifest + Segment persistence on `object_store`.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};

use crc32fast::Hasher;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload, WriteMultipart};
use object_store::path::Path as StorePath;

use crate::{Metric, RabitqError, RotatorType};

pub const V4_MANIFEST_MAGIC: [u8; 4] = *b"RBQ3";
pub const V4_MANIFEST_VERSION: u32 = 1;
pub const V4_SEGMENT_MAGIC: [u8; 4] = *b"SEG1";
pub const MANIFEST_FILENAME: &str = "manifest.bin";

// ---- little-endian read/write with optional hasher ----

macro_rules! rle {
    ($r:expr, u8)  => {{ let mut b=[0u8;1]; $r.read_exact(&mut b)?; b[0] }};
    ($r:expr, u32) => {{ let mut b=[0u8;4]; $r.read_exact(&mut b)?; u32::from_le_bytes(b) }};
    ($r:expr, u64) => {{ let mut b=[0u8;8]; $r.read_exact(&mut b)?; u64::from_le_bytes(b) }};
    ($r:expr, f32) => {{ let mut b=[0u8;4]; $r.read_exact(&mut b)?; f32::from_le_bytes(b) }};
}

macro_rules! wle {
    ($w:expr, $v:expr, u8)  => { $w.write_all(&[$v as u8]).unwrap(); };
    ($w:expr, $v:expr, u32) => { $w.write_all(&($v as u32).to_le_bytes()).unwrap(); };
    ($w:expr, $v:expr, u64) => { $w.write_all(&($v as u64).to_le_bytes()).unwrap(); };
    ($w:expr, $v:expr, f32) => { $w.write_all(&($v).to_le_bytes()).unwrap(); };
}

macro_rules! hup {
    ($h:expr, $d:expr) => { if let Some(h) = $h { h.update($d); } };
}

// ---- conversions ----

fn u2u64(v: usize) -> Result<u64, RabitqError> {
    u64::try_from(v).map_err(|_| RabitqError::InvalidPersistence("usize exceeds u64"))
}
fn uf64(v: u64) -> Result<usize, RabitqError> {
    usize::try_from(v).map_err(|_| RabitqError::InvalidPersistence("value exceeds usize"))
}
fn mt(m: Metric) -> u8 { match m { Metric::L2 => 0, Metric::InnerProduct => 1 } }
fn tm(tag: u8) -> Option<Metric> { match tag { 0 => Some(Metric::L2), 1 => Some(Metric::InnerProduct), _ => None } }

fn os_err(e: object_store::Error) -> RabitqError {
    RabitqError::Io(std::io::Error::other(e.to_string()))
}

// ---- types ----

#[derive(Debug, Clone)]
pub struct ClusterManifestEntry {
    pub cluster_id: u32,
    pub segment_filename: String,
    pub segment_version: u32,
    pub num_vectors: u32,
    pub file_size: u64,
}

#[derive(Debug, Clone)]
pub struct ManifestHeader {
    pub dim: usize, pub padded_dim: usize, pub metric: Metric,
    pub rotator_type: RotatorType, pub rotator_data: Vec<u8>,
    pub ex_bits: usize, pub total_bits: usize,
}

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

// ---- manifest read/write ----

pub async fn load_manifest(
    store: &dyn ObjectStore,
) -> Result<(ManifestHeader, BTreeMap<u32, ClusterManifestEntry>), RabitqError> {
    let key = StorePath::from(MANIFEST_FILENAME);
    let result = store.get(&key).await.map_err(os_err)?;
    let bytes = result.bytes().await.map_err(os_err)?;
    let mut r = Cursor::new(bytes.as_ref());

    let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
    if magic != V4_MANIFEST_MAGIC { return Err(RabitqError::InvalidPersistence("not a V4 manifest")); }
    let version = rle!(r, u32);
    if version != V4_MANIFEST_VERSION { return Err(RabitqError::InvalidPersistence("unsupported manifest version")); }

    let mut h = Hasher::new();
    let dim = rle!(r, u32) as usize;   hup!(Some(&mut h), &(dim as u32).to_le_bytes());
    let pd = rle!(r, u32) as usize;    hup!(Some(&mut h), &(pd as u32).to_le_bytes());
    let mtag = rle!(r, u8);            hup!(Some(&mut h), &[mtag]);
    let rtag = rle!(r, u8);            hup!(Some(&mut h), &[rtag]);
    let eb = rle!(r, u8) as usize;     hup!(Some(&mut h), &[eb as u8]);
    let tb = rle!(r, u8) as usize;     hup!(Some(&mut h), &[tb as u8]);
    let _tv = rle!(r, u64);            hup!(Some(&mut h), &0u64.to_le_bytes()); // placeholder
    let rdl = uf64(rle!(r, u64))?;     hup!(Some(&mut h), &(rdl as u64).to_le_bytes()); // already fine, just confirming
    let mut rd = vec![0u8; rdl]; r.read_exact(&mut rd)?; h.update(&rd);

    let cc = rle!(r, u32) as usize;    hup!(Some(&mut h), &(cc as u32).to_le_bytes());
    let mut map: BTreeMap<u32, ClusterManifestEntry> = BTreeMap::new();
    for _ in 0..cc {
        let cid = rle!(r, u32);        hup!(Some(&mut h), &cid.to_le_bytes());
        let sv = rle!(r, u32);         hup!(Some(&mut h), &sv.to_le_bytes());
        let nv = rle!(r, u32);         hup!(Some(&mut h), &nv.to_le_bytes());
        let fs = rle!(r, u64);         hup!(Some(&mut h), &fs.to_le_bytes());
        let fl = rle!(r, u32) as usize; hup!(Some(&mut h), &(fl as u32).to_le_bytes());
        let mut fb = vec![0u8; fl]; r.read_exact(&mut fb)?; h.update(&fb);
        let fname = String::from_utf8(fb).map_err(|_| RabitqError::InvalidPersistence("non-UTF8 filename"))?;
        map.insert(cid, ClusterManifestEntry { cluster_id:cid, segment_filename:fname, segment_version:sv, num_vectors:nv, file_size:fs });
    }
    let computed = h.finalize();
    let stored = rle!(r, u32);
    if computed != stored { return Err(RabitqError::InvalidPersistence("manifest checksum mismatch")); }

    let metric = tm(mtag).ok_or(RabitqError::InvalidPersistence("unknown metric tag"))?;
    let rotator_type = RotatorType::from_u8(rtag).ok_or(RabitqError::InvalidPersistence("unknown rotator type"))?;
    Ok((ManifestHeader { dim, padded_dim: pd, metric, rotator_type, rotator_data: rd, ex_bits: eb, total_bits: tb }, map))
}

pub async fn save_manifest(
    store: &dyn ObjectStore,
    header: &ManifestHeader,
    cluster_map: &BTreeMap<u32, ClusterManifestEntry>,
) -> Result<(), RabitqError> {
    let mut b = Vec::new();
    b.write_all(&V4_MANIFEST_MAGIC).unwrap();
    wle!(b, V4_MANIFEST_VERSION, u32);

    let mut h = Hasher::new();
    wle!(b, header.dim, u32);    hup!(Some(&mut h), &(header.dim as u32).to_le_bytes());
    wle!(b, header.padded_dim, u32); hup!(Some(&mut h), &(header.padded_dim as u32).to_le_bytes());
    wle!(b, mt(header.metric), u8); hup!(Some(&mut h), &[mt(header.metric)]);
    wle!(b, header.rotator_type as u8, u8); hup!(Some(&mut h), &[header.rotator_type as u8]);
    wle!(b, header.ex_bits, u8); hup!(Some(&mut h), &[header.ex_bits as u8]);
    wle!(b, header.total_bits, u8); hup!(Some(&mut h), &[header.total_bits as u8]);
    wle!(b, 0u64, u64); hup!(Some(&mut h), &0u64.to_le_bytes());
    wle!(b, header.rotator_data.len(), u64); hup!(Some(&mut h), &(header.rotator_data.len() as u64).to_le_bytes());
    b.write_all(&header.rotator_data).unwrap(); h.update(&header.rotator_data);
    wle!(b, cluster_map.len(), u32); hup!(Some(&mut h), &(cluster_map.len() as u32).to_le_bytes());

    for e in cluster_map.values() {
        wle!(b, e.cluster_id, u32);      hup!(Some(&mut h), &e.cluster_id.to_le_bytes());
        wle!(b, e.segment_version, u32); hup!(Some(&mut h), &e.segment_version.to_le_bytes());
        wle!(b, e.num_vectors, u32);     hup!(Some(&mut h), &e.num_vectors.to_le_bytes());
        wle!(b, e.file_size, u64);       hup!(Some(&mut h), &e.file_size.to_le_bytes());
        let fb = e.segment_filename.as_bytes();
        wle!(b, fb.len(), u32); hup!(Some(&mut h), &(fb.len() as u32).to_le_bytes());
        b.write_all(fb).unwrap(); h.update(fb);
    }
    wle!(b, h.finalize(), u32);

    let key = StorePath::from(MANIFEST_FILENAME);
    store.put(&key, PutPayload::from_bytes(b.into())).await.map_err(os_err)?;
    Ok(())
}

// ---- segment read (full) ----

pub async fn read_segment_full(
    store: &dyn ObjectStore, key: &str,
) -> Result<ClusterSegmentData, RabitqError> {
    let result = store.get(&StorePath::from(key)).await.map_err(os_err)?;
    let bytes = result.bytes().await.map_err(os_err)?;
    let mut r = Cursor::new(bytes.as_ref());

    let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
    if magic != V4_SEGMENT_MAGIC { return Err(RabitqError::InvalidPersistence("not a V4 segment")); }
    let mut h = Hasher::new();

    let cluster_id = rle!(r, u32); hup!(Some(&mut h), &cluster_id.to_le_bytes());
    let _sv = rle!(r, u32);       hup!(Some(&mut h), &_sv.to_le_bytes());
    let pd = rle!(r, u32) as usize; hup!(Some(&mut h), &(pd as u32).to_le_bytes());
    let eb = rle!(r, u8) as usize;  hup!(Some(&mut h), &[eb as u8]);
    let nv = rle!(r, u32) as usize; hup!(Some(&mut h), &(nv as u32).to_le_bytes());

    let mut centroid = vec![0.0f32; pd];
    for v in &mut centroid { *v = rle!(r, f32); hup!(Some(&mut h), &v.to_le_bytes()); }

    let mut ids = Vec::with_capacity(nv);
    for _ in 0..nv { let id = rle!(r, u64); hup!(Some(&mut h), &id.to_le_bytes()); ids.push(uf64(id)?); }

    let bdl = uf64(rle!(r, u64))?; hup!(Some(&mut h), &(bdl as u64).to_le_bytes());
    let mut batch_data = vec![0u8; bdl]; r.read_exact(&mut batch_data)?; h.update(&batch_data);

    let ec = rle!(r, u32) as usize; hup!(Some(&mut h), &(ec as u32).to_le_bytes());
    let mut ex_codes_packed = Vec::with_capacity(ec);
    for _ in 0..ec {
        let el = uf64(rle!(r, u64))?; hup!(Some(&mut h), &(el as u64).to_le_bytes());
        let mut d = vec![0u8; el]; r.read_exact(&mut d)?; h.update(&d);
        ex_codes_packed.push(d);
    }

    let mut f_add_ex = Vec::with_capacity(nv);
    for _ in 0..nv { let v = rle!(r, f32); hup!(Some(&mut h), &v.to_le_bytes()); f_add_ex.push(v); }
    let mut f_rescale_ex = Vec::with_capacity(nv);
    for _ in 0..nv { let v = rle!(r, f32); hup!(Some(&mut h), &v.to_le_bytes()); f_rescale_ex.push(v); }
    let mut delta = Vec::with_capacity(nv);
    for _ in 0..nv { let v = rle!(r, f32); hup!(Some(&mut h), &v.to_le_bytes()); delta.push(v); }
    let mut vl = Vec::with_capacity(nv);
    for _ in 0..nv { let v = rle!(r, f32); hup!(Some(&mut h), &v.to_le_bytes()); vl.push(v); }

    let computed = h.finalize();
    let stored = rle!(r, u32);
    if computed != stored { return Err(RabitqError::InvalidPersistence("segment checksum mismatch")); }

    Ok(ClusterSegmentData { cluster_id, centroid, padded_dim: pd, ex_bits: eb, ids, batch_data, ex_codes_packed, f_add_ex, f_rescale_ex, delta, vl })
}

// ---- segment read (centroid only) ----

pub async fn read_segment_centroid(
    store: &dyn ObjectStore, key: &str, padded_dim: usize,
) -> Result<(u32, Vec<f32>), RabitqError> {
    let result = store.get(&StorePath::from(key)).await.map_err(os_err)?;
    let bytes = result.bytes().await.map_err(os_err)?;
    let b = bytes.as_ref();
    if b.len() < 21 + padded_dim * 4 { return Err(RabitqError::InvalidPersistence("segment too short")); }
    let cluster_id = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let mut centroid = vec![0.0f32; padded_dim];
    for i in 0..padded_dim {
        let off = 21 + i * 4;
        centroid[i] = f32::from_le_bytes([b[off], b[off+1], b[off+2], b[off+3]]);
    }
    Ok((cluster_id, centroid))
}

// ---- segment write ----

pub async fn write_segment(
    store: &dyn ObjectStore, key: &str, seg: &ClusterSegmentData, version: u32,
) -> Result<u64, RabitqError> {
    let upload = store.put_multipart(&StorePath::from(key)).await.map_err(os_err)?;
    let mut w = WriteMultipart::new(upload);
    let mut h = Hasher::new();
    let mut b = Vec::new();

    b.write_all(&V4_SEGMENT_MAGIC).unwrap();
    wle!(b, seg.cluster_id, u32);  hup!(Some(&mut h), &seg.cluster_id.to_le_bytes());
    wle!(b, version, u32);         hup!(Some(&mut h), &version.to_le_bytes());
    wle!(b, seg.padded_dim, u32);  hup!(Some(&mut h), &(seg.padded_dim as u32).to_le_bytes());
    wle!(b, seg.ex_bits, u8);      hup!(Some(&mut h), &[seg.ex_bits as u8]);
    wle!(b, seg.ids.len(), u32);   hup!(Some(&mut h), &(seg.ids.len() as u32).to_le_bytes());

    for &v in &seg.centroid { wle!(b, v, f32); hup!(Some(&mut h), &v.to_le_bytes()); }
    for &id in &seg.ids { wle!(b, id, u64); hup!(Some(&mut h), &id.to_le_bytes()); }

    wle!(b, seg.batch_data.len(), u64); hup!(Some(&mut h), &(seg.batch_data.len() as u64).to_le_bytes());
    b.write_all(&seg.batch_data).unwrap(); h.update(&seg.batch_data);

    wle!(b, seg.ex_codes_packed.len(), u32); hup!(Some(&mut h), &(seg.ex_codes_packed.len() as u32).to_le_bytes());
    for ex in &seg.ex_codes_packed {
        wle!(b, ex.len(), u64); hup!(Some(&mut h), &(ex.len() as u64).to_le_bytes());
        b.write_all(ex).unwrap(); h.update(ex);
    }

    for &v in &seg.f_add_ex { wle!(b, v, f32); hup!(Some(&mut h), &v.to_le_bytes()); }
    for &v in &seg.f_rescale_ex { wle!(b, v, f32); hup!(Some(&mut h), &v.to_le_bytes()); }
    for &v in &seg.delta { wle!(b, v, f32); hup!(Some(&mut h), &v.to_le_bytes()); }
    for &v in &seg.vl { wle!(b, v, f32); hup!(Some(&mut h), &v.to_le_bytes()); }

    wle!(b, h.finalize(), u32);

    let file_size = b.len() as u64;
    w.write(&b);
    w.finish().await.map_err(os_err)?;
    Ok(file_size)
}

// ---- helpers ----

pub fn segment_filename(cluster_id: u32, version: u32) -> String {
    format!("cluster_{cluster_id:04}_{version:04}.seg")
}

pub async fn delete_segment(store: &dyn ObjectStore, key: &str) -> Result<(), RabitqError> {
    store.delete(&StorePath::from(key)).await.map_err(os_err)
}

pub async fn manifest_exists(store: &dyn ObjectStore) -> bool {
    store.head(&StorePath::from(MANIFEST_FILENAME)).await.is_ok()
}
