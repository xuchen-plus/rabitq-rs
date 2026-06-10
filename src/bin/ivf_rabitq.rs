/// Build / insert IVF+RaBitQ index (V3 single-file or V4 object_store).
///
/// Usage (V3):
///   cargo run --release --bin ivf_rabitq -- --base ... --nlist 4096 --bits 7 --save index.bin
///
/// Usage (V4, local filesystem):
///   cargo run --release --bin ivf_rabitq -- --base ... --nlist 4096 --bits 7 --format v4 --save /tmp/v4/
///
/// Usage (V4, S3):
///   cargo run --release --bin ivf_rabitq -- --base ... --nlist 128 --bits 7 --format v4 \
///       --store s3 --s3-endpoint http://localhost:9000 --s3-bucket rabitq-test \
///       --s3-access-key minioadmin1 --s3-secret-key minioadmin1 \
///       --save /prefix/v4/ --limit 100000
///
/// Usage (V4, incremental insert via S3):
///   cargo run --release --bin ivf_rabitq -- --load /prefix/v4/ --format v4 \
///       --store s3 --s3-endpoint http://localhost:9000 --s3-bucket rabitq-test \
///       --s3-access-key minioadmin1 --s3-secret-key minioadmin1 \
///       --insert new.fvecs --save /prefix/v4/

use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;

use rabitq_rs::io::{read_fvecs, read_ids};
use rabitq_rs::{IvfRabitqBuilder, IvfRabitqIndex, Metric, RotatorType};

#[derive(Debug, Clone, Copy, PartialEq)]
enum SaveFormat { V3, V4 }

#[derive(Debug, Clone, Copy, PartialEq)]
enum StoreType { Fs, S3 }

#[derive(Debug)]
struct Args {
    base: Option<PathBuf>,
    centroids: Option<PathBuf>,
    assignments: Option<PathBuf>,
    nlist: Option<usize>,
    bits: usize,
    save: Option<PathBuf>,
    load: Option<PathBuf>,
    insert: Option<PathBuf>,
    format: SaveFormat,
    faster_config: bool,
    metric: Metric,
    seed: u64,
    limit: Option<usize>,
    stream: bool,
    stream_batch_size: usize,
    // S3
    store_type: StoreType,
    s3_endpoint: String,
    s3_bucket: String,
    s3_access_key: String,
    s3_secret_key: String,
    s3_allow_http: bool,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let mut base = None; let mut centroids = None; let mut assignments = None;
    let mut nlist = None; let mut bits = 7usize; let mut save = None;
    let mut load = None; let mut insert = None;
    let mut format = SaveFormat::V3; let mut faster_config = false;
    let mut metric = Metric::L2; let mut seed = 42u64; let mut limit = None;
    let mut stream = false; let mut stream_batch_size = 100_000usize;
    let mut store_type = StoreType::Fs;
    let mut s3_endpoint = "http://localhost:9000".to_string();
    let mut s3_bucket = String::new();
    let mut s3_access_key = "minioadmin1".to_string();
    let mut s3_secret_key = "minioadmin1".to_string();
    let mut s3_allow_http = true;

    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--base" => { i+=1; base = Some(PathBuf::from(&a[i])); }
            "--centroids" => { i+=1; centroids = Some(PathBuf::from(&a[i])); }
            "--assignments" => { i+=1; assignments = Some(PathBuf::from(&a[i])); }
            "--nlist" => { i+=1; nlist = Some(a[i].parse()?); }
            "--bits" => { i+=1; bits = a[i].parse()?; }
            "--save" => { i+=1; save = Some(PathBuf::from(&a[i])); }
            "--load" => { i+=1; load = Some(PathBuf::from(&a[i])); }
            "--insert" => { i+=1; insert = Some(PathBuf::from(&a[i])); }
            "--format" => { i+=1; format = match a[i].as_str() { "v3"=>SaveFormat::V3, "v4"=>SaveFormat::V4, o=>{eprintln!("unknown format: {o}");std::process::exit(1);}}; }
            "--faster-config" => { faster_config = true; }
            "--ip" => { metric = Metric::InnerProduct; }
            "--seed" => { i+=1; seed = a[i].parse()?; }
            "--limit" => { i+=1; limit = Some(a[i].parse()?); }
            "--stream" => { stream = true; }
            "--stream-batch-size" => { i+=1; stream_batch_size = a[i].parse()?; }
            "--store" => { i+=1; store_type = match a[i].as_str() { "fs"=>StoreType::Fs, "s3"=>StoreType::S3, o=>{eprintln!("unknown store: {o}");std::process::exit(1);}}; }
            "--s3-endpoint" => { i+=1; s3_endpoint = a[i].clone(); }
            "--s3-bucket" => { i+=1; s3_bucket = a[i].clone(); }
            "--s3-access-key" => { i+=1; s3_access_key = a[i].clone(); }
            "--s3-secret-key" => { i+=1; s3_secret_key = a[i].clone(); }
            "--s3-allow-http" => { s3_allow_http = true; }
            other => { eprintln!("unknown argument: {other}"); std::process::exit(1); }
        }
        i += 1;
    }
    if load.is_none() {
        if base.is_none() { return Err("--base required".into()); }
        if save.is_none() { return Err("--save required".into()); }
    }
    if store_type == StoreType::S3 && s3_bucket.is_empty() {
        return Err("--s3-bucket required for S3 store".into());
    }
    Ok(Args { base, centroids, assignments, nlist, bits, save, load, insert, format, faster_config, metric, seed, limit, stream, stream_batch_size, store_type, s3_endpoint, s3_bucket, s3_access_key, s3_secret_key, s3_allow_http })
}

// ---- Store creation ----

fn create_store(args: &Args, prefix: &str) -> Arc<dyn ObjectStore> {
    match args.store_type {
        StoreType::Fs => {
            let _ = std::fs::create_dir_all(prefix);
            Arc::new(LocalFileSystem::new_with_prefix(prefix).expect("LocalFileSystem"))
        }
        StoreType::S3 => {
            use object_store::aws::AmazonS3Builder;
            use object_store::prefix::PrefixStore;

            let s3 = AmazonS3Builder::new()
                .with_endpoint(&args.s3_endpoint)
                .with_access_key_id(&args.s3_access_key)
                .with_secret_access_key(&args.s3_secret_key)
                .with_bucket_name(&args.s3_bucket)
                .with_allow_http(args.s3_allow_http)
                .build()
                .expect("Failed to create S3 store");
            Arc::new(PrefixStore::new(s3, prefix))
        }
    }
}

// Metrics tracked across operations.
struct IoMetrics {
    bytes_read: u64,
    bytes_written: u64,
}

fn fmt_bytes(n: u64) -> String {
    if n < 1024 { format!("{n} B") }
    else if n < 1024*1024 { format!("{:.1} KB", n as f64 / 1024.0) }
    else { format!("{:.1} MB", n as f64 / (1024.0*1024.0)) }
}

/// Truly streaming fvecs batch reader.  Holds an open file and reads
/// `batch_size` vectors at a time on each call to `next_batch()`.
/// Only one batch is in memory at any time.
struct BatchReader {
    reader: BufReader<std::fs::File>,
    dim: usize,
    batch_size: usize,
    remaining: usize,
    buf: Vec<f32>,
}

impl BatchReader {
    fn new(path: &Path, dim: usize, batch_size: usize, limit: Option<usize>) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(batch_size * 4 * 16, file);
        let remaining = limit.unwrap_or(usize::MAX);
        Ok(Self {
            reader, dim, batch_size, remaining,
            buf: Vec::with_capacity(batch_size * dim),
        })
    }

    /// Read the next batch, returning `None` when exhausted.
    fn next_batch(&mut self) -> Option<Vec<f32>> {
        if self.remaining == 0 { return None; }
        self.buf.clear();
        let mut batch_n: usize = 0;
        while batch_n < self.batch_size && self.remaining > 0 {
            let mut hdr = [0u8; 4];
            if self.reader.read_exact(&mut hdr).is_err() {
                break; // EOF
            }
            let d = u32::from_le_bytes(hdr) as usize;
            let start = self.buf.len();
            self.buf.resize(start + self.dim, 0.0f32);
            let to_read = d.min(self.dim);
            for i in 0..to_read {
                let mut vb = [0u8; 4];
                if self.reader.read_exact(&mut vb).is_err() { break; }
                self.buf[start + i] = f32::from_le_bytes(vb);
            }
            if d > self.dim {
                let _ = self.reader.seek(SeekFrom::Current(((d - self.dim) * 4) as i64));
            }
            batch_n += 1;
            self.remaining -= 1;
        }
        if batch_n == 0 { return None; }
        self.buf.truncate(batch_n * self.dim);
        Some(std::mem::take(&mut self.buf))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RAYON_NUM_THREADS").is_err() { std::env::set_var("RAYON_NUM_THREADS", "16"); }
    let args = parse_args()?;

    let rt = tokio::runtime::Runtime::new()?;

    // ── Load + insert path ──
    if let Some(ref load_path) = args.load {
        println!("=== RaBitQ IVF Index (load + insert) ===");
        println!("Load: {}  Format: {:?}  Store: {:?}", load_path.display(), args.format, args.store_type);
        let t0 = Instant::now();

        if args.format == SaveFormat::V4 {
            let load_prefix = load_path.to_string_lossy().to_string();
            let store = create_store(&args, &load_prefix);
            let mut load_metrics = IoMetrics { bytes_read: 0, bytes_written: 0 };

            // Read insert file header to get dim (params ignored in Loaded mode).
            let dim = if let Some(ref ip) = args.insert {
                let preview = read_fvecs(ip, Some(1))?;
                preview.first().map_or(0, |v| v.len())
            } else {
                0
            };
            let nlist = args.nlist.unwrap_or(0);
            let bits = args.bits;

            let mut builder = rt.block_on(IvfRabitqBuilder::load(
                store.clone(),
                dim, nlist, bits,
                args.metric,
                RotatorType::FhtKacRotator,
                args.seed,
                args.faster_config,
            ))?;

            // Compute read metrics
            let cluster_count = builder.cluster_count();
            let pd = builder.padded_dim();
            let centroid_bytes_per_seg = (21 + pd * 4) as u64;
            // manifest read (approximate: header ~200B + ~45B per cluster entry)
            let manifest_read = 200 + cluster_count as u64 * 45;
            let centroid_reads = cluster_count as u64 * centroid_bytes_per_seg;
            load_metrics.bytes_read = manifest_read + centroid_reads;

            println!("  Loaded in {:.1}s  (centroids only)", t0.elapsed().as_secs_f64());
            println!("  [IO] Read: {}  (manifest={} + {} centroids × {})",
                     fmt_bytes(load_metrics.bytes_read),
                     fmt_bytes(manifest_read),
                     cluster_count,
                     fmt_bytes(centroid_bytes_per_seg));

            if let Some(ref ip) = args.insert {
                println!("Inserting vectors from {}...", ip.display());
                let all_new = read_fvecs(ip, None)?;
                if !all_new.is_empty() {
                    let dim = all_new[0].len();
                    let batch_size = 50_000;
                    let mut total_inserted: usize = 0;
                    let t0 = Instant::now();
                    for chunk in all_new.chunks(batch_size) {
                        let mut flat = Vec::with_capacity(chunk.len() * dim);
                        for v in chunk { flat.extend_from_slice(v); }
                        builder.insert_batch(flat)?;
                        total_inserted += chunk.len();
                    }
                    println!("  Inserted {} vectors in {:.1}s ({:.0} vec/s)",
                             total_inserted, t0.elapsed().as_secs_f64(),
                             total_inserted as f64 / t0.elapsed().as_secs_f64());
                }
            }

            if let Some(ref sp) = args.save {
                println!("Saving (incremental / delta flush) to {}...", sp.display());
                let t0 = Instant::now();
                let save_prefix = sp.to_string_lossy().to_string();
                let save_store = create_store(&args, &save_prefix);

                // Track write metrics: we'll print total after flush.
                let index = rt.block_on(builder.flush(&*save_store))?;

                let elapsed = t0.elapsed().as_secs_f64();
                println!("  Saved in {:.1}s  ({:.1} MB)", elapsed, index.estimate_memory_mb());
                println!("  [IO] Write: delta-segment bytes written (see Flush log above for per-segment sizes)");
            }
            return Ok(());
        }

        // V3: traditional load/insert/save path
        let index = IvfRabitqIndex::load_from_path(load_path)?;
        println!("  Loaded in {:.1}s  ({} vectors, {} clusters, {:.1} MB)",
                 t0.elapsed().as_secs_f64(), index.len(), index.cluster_count(), index.estimate_memory_mb());
        if let Some(ref ip) = args.insert {
            println!("Inserting vectors from {}...", ip.display());
            let dim = index.dim();
            let all_new = read_fvecs(ip, None)?;
            if !all_new.is_empty() {
                let batch_size = 50_000;
                let mut start_id = index.len() as u64;
                let mut index = index;
                let t0 = Instant::now();
                for chunk in all_new.chunks(batch_size) {
                    let mut flat = Vec::with_capacity(chunk.len() * dim);
                    for v in chunk { flat.extend_from_slice(v); }
                    let n = index.insert_batch(start_id, flat)?;
                    start_id += n as u64;
                }
                index.flush_all_pending();
                println!("  Inserted {} vectors in {:.1}s ({:.0} vec/s)",
                         all_new.len(), t0.elapsed().as_secs_f64(), all_new.len() as f64 / t0.elapsed().as_secs_f64());
                println!("  New total: {} vectors, {:.1} MB", index.len(), index.estimate_memory_mb());
                if let Some(ref sp) = args.save {
                    println!("Saving to {}...", sp.display());
                    let t0 = Instant::now();
                    index.save_to_path(sp)?;
                    println!("  Saved in {:.1}s", t0.elapsed().as_secs_f64());
                }
            }
        }
        return Ok(());
    }

    // ── Build path ──
    let base = args.base.as_ref().unwrap();
    let save_path = args.save.as_ref().unwrap();
    println!("=== RaBitQ IVF Index Builder ===");
    println!("Base: {}  Bits: {}  Metric: {:?}  Format: {:?}  Store: {:?}  Save: {}",
             base.display(), args.bits, args.metric, args.format, args.store_type, save_path.display());

    // Read just 1 vector to get dim (free immediately).
    let dim = {
        let preview = read_fvecs(base, Some(1))?;
        if preview.is_empty() { return Err("no vectors loaded".into()); }
        preview[0].len()
    };
    println!("Dim: {}  Limit: {:?}", dim, args.limit);

    let total_start = Instant::now();
    let index = if args.format == SaveFormat::V4 {
        // ── V4: always use streaming builder (low memory) ──
        let nlist = args.nlist.ok_or("--nlist required")?;
        let batch_size = args.stream_batch_size;
        let save_prefix = save_path.to_string_lossy().to_string();
        let store = create_store(&args, &save_prefix);
        let mut builder = rt.block_on(IvfRabitqBuilder::load(
            store, dim, nlist, args.bits, args.metric,
            RotatorType::FhtKacRotator, args.seed, args.faster_config,
        ))?;

        // Phase 1: reservoir sample (streaming — one batch at a time)
        println!("Phase 1: reservoir sampling ({} clusters, batch_size={})...", nlist, batch_size);
        let t0 = Instant::now();
        let mut seen: usize = 0;
        {
            let mut rdr = BatchReader::new(base, dim, batch_size, args.limit)?;
            while let Some(batch) = rdr.next_batch() {
                seen += batch.len() / dim;
                builder.insert_batch(batch)?;
            }
        }
        println!("  Reservoir: processed {} vectors in {:.1}s", seen, t0.elapsed().as_secs_f64());

        // Phase 2: build (streaming — one batch at a time via callback)
        println!("Phase 2: building (streaming rotation + quantization)...");
        let mut rdr2 = BatchReader::new(base, dim, batch_size, args.limit)?;
        builder.build(Some(&mut || rdr2.next_batch()))?
    } else if let (Some(cp), Some(ap)) = (&args.centroids, &args.assignments) {
        // ── V3: pre-clustered ──
        let vectors = read_fvecs(base, args.limit)?;
        let centroids = read_fvecs(cp, None)?;
        let assignments = read_ids(ap, None)?;
        IvfRabitqIndex::train_with_clusters(&vectors, &centroids, &assignments,
            args.bits, args.metric, RotatorType::FhtKacRotator, args.seed, args.faster_config)?
    } else {
        // ── V3: full in-memory training ──
        let vectors = read_fvecs(base, args.limit)?;
        let nlist = args.nlist.ok_or("--nlist required")?;
        IvfRabitqIndex::train(&vectors, nlist, args.bits, args.metric,
            RotatorType::FhtKacRotator, args.seed, args.faster_config)?
    };

    println!("\nIndex built in {:.1}s  Vectors: {}  Clusters: {}  Memory: {:.1} MB",
             total_start.elapsed().as_secs_f64(), index.len(), index.cluster_count(), index.estimate_memory_mb());

    println!("\nSaving to {}...", save_path.display());
    let t0 = Instant::now();
    match args.format {
        SaveFormat::V3 => index.save_to_path(save_path)?,
        SaveFormat::V4 => {
            let save_prefix = save_path.to_string_lossy().to_string();
            let save_store = create_store(&args, &save_prefix);
            rt.block_on(index.save_to_v4(&*save_store))?;

            // Print IO metrics
            let cluster_count = index.cluster_count();
            let total_vectors = index.len();
            let pd = index.padded_dim();
            let avg_nv = if cluster_count > 0 { total_vectors as u64 / cluster_count as u64 } else { 0 };
            let avg_seg: u64 = 21 + pd as u64 * 4 + avg_nv * 8 + (avg_nv+31)/32 * 4224 + avg_nv * 736 + 4;
            let total_base_segment_bytes = cluster_count as u64 * avg_seg;
            let manifest_write: u64 = 200 + cluster_count as u64 * 45;
            let total_write = total_base_segment_bytes + manifest_write;
            println!("  [IO] Write: {}  ({} base segments, manifest)",
                     fmt_bytes(total_write), cluster_count);
        }
    }
    println!("  Saved in {:.1}s", t0.elapsed().as_secs_f64());
    println!("Done. Total: {:.1}s", total_start.elapsed().as_secs_f64());
    Ok(())
}
