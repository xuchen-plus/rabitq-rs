/// Build / insert IVF+RaBitQ index (V3 single-file or V4 object_store).
///
/// Usage (V3):
///   cargo run --release --bin ivf_rabitq -- --base ... --nlist 4096 --bits 7 --save index.bin
///
/// Usage (V4, local filesystem):
///   cargo run --release --bin ivf_rabitq -- --base ... --nlist 4096 --bits 7 --format v4 --save /tmp/v4/
///
/// Usage (V4, incremental insert):
///   cargo run --release --bin ivf_rabitq -- --load /tmp/v4/ --format v4 --insert new.fvecs --save /tmp/v4/

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;

use rabitq_rs::io::{read_fvecs, read_fvecs_from_reader, read_ids};
use rabitq_rs::{IvfRabitqBuilder, IvfRabitqIndex, Metric, RotatorType};

#[derive(Debug, Clone, Copy, PartialEq)]
enum SaveFormat { V3, V4 }

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
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let mut base = None; let mut centroids = None; let mut assignments = None;
    let mut nlist = None; let mut bits = 7usize; let mut save = None;
    let mut load = None; let mut insert = None;
    let mut format = SaveFormat::V3; let mut faster_config = false;
    let mut metric = Metric::L2; let mut seed = 42u64; let mut limit = None;
    let mut stream = false; let mut stream_batch_size = 100_000usize;

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
            other => { eprintln!("unknown argument: {other}"); std::process::exit(1); }
        }
        i += 1;
    }
    if load.is_none() {
        if base.is_none() { return Err("--base required".into()); }
        if save.is_none() { return Err("--save required".into()); }
    }
    Ok(Args { base, centroids, assignments, nlist, bits, save, load, insert, format, faster_config, metric, seed, limit, stream, stream_batch_size })
}

// ---- V4 helpers ----

fn v4_store(dir: &Path) -> Arc<dyn ObjectStore> {
    let _ = std::fs::create_dir_all(dir);
    Arc::new(LocalFileSystem::new_with_prefix(dir).expect("LocalFileSystem"))
}

fn v4_load_index(dir: &Path) -> Result<IvfRabitqIndex, Box<dyn std::error::Error>> {
    let store = v4_store(dir);
    let rt = tokio::runtime::Runtime::new()?;
    Ok(rt.block_on(IvfRabitqIndex::load_from_v4(store))?)
}

fn v4_save_index(index: &IvfRabitqIndex, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let store = v4_store(dir);
    let rt = tokio::runtime::Runtime::new()?;
    Ok(rt.block_on(index.save_to_v4(&*store))?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RAYON_NUM_THREADS").is_err() { std::env::set_var("RAYON_NUM_THREADS", "16"); }
    let args = parse_args()?;

    // ── Load + insert path ──
    if let Some(ref load_path) = args.load {
        println!("=== RaBitQ IVF Index (load + insert) ===");
        println!("Load: {}  Format: {:?}", load_path.display(), args.format);
        let t0 = Instant::now();
        let index = match args.format {
            SaveFormat::V3 => IvfRabitqIndex::load_from_path(load_path)?,
            SaveFormat::V4 => v4_load_index(load_path)?,
        };
        println!("  Loaded in {:.1}s  ({} vectors, {} clusters, {:.1} MB)",
                 t0.elapsed().as_secs_f64(), index.len(), index.cluster_count(), index.estimate_memory_mb());
        if let Some(ref ip) = args.insert {
            println!("Inserting vectors from {}...", ip.display());
            let dim = index.dim();
            let all_new = read_fvecs(ip, None)?;
            if !all_new.is_empty() {
                let batch_size = 50_000;
                let mut start_id = index.len();
                let mut index = index;
                let t0 = Instant::now();
                for chunk in all_new.chunks(batch_size) {
                    let mut flat = Vec::with_capacity(chunk.len() * dim);
                    for v in chunk { flat.extend_from_slice(v); }
                    let n = index.insert_batch(start_id, flat)?;
                    start_id += n;
                }
                index.flush_all_pending();
                println!("  Inserted {} vectors in {:.1}s ({:.0} vec/s)",
                         all_new.len(), t0.elapsed().as_secs_f64(), all_new.len() as f64 / t0.elapsed().as_secs_f64());
                println!("  New total: {} vectors, {:.1} MB", index.len(), index.estimate_memory_mb());
                if let Some(ref sp) = args.save {
                    println!("Saving to {}...", sp.display());
                    let t0 = Instant::now();
                    match args.format {
                        SaveFormat::V3 => index.save_to_path(sp)?,
                        SaveFormat::V4 => v4_save_index(&index, sp)?,
                    }
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
    println!("Base: {}  Bits: {}  Metric: {:?}  Format: {:?}  Save: {}",
             base.display(), args.bits, args.metric, args.format, save_path.display());

    println!("Loading vectors from {}...", base.display());
    let t0 = Instant::now();
    let vectors = read_fvecs(base, args.limit)?;
    println!("  Loaded {} vectors in {:.1}s ({:.1} MB)", vectors.len(), t0.elapsed().as_secs_f64(),
             vectors.len() * vectors[0].len() * 4 / (1024*1024));
    if vectors.is_empty() { return Err("no vectors loaded".into()); }
    let dim = vectors[0].len();

    let total_start = Instant::now();
    let index = if args.stream && args.format == SaveFormat::V4 {
        let nlist = args.nlist.ok_or("--nlist required")?;
        let batch_size = args.stream_batch_size;
        drop(vectors);
        let rt = tokio::runtime::Runtime::new()?;
        let store = v4_store(save_path);
        let mut builder = rt.block_on(IvfRabitqBuilder::load(
            store, dim, nlist, args.bits, args.metric, RotatorType::FhtKacRotator, args.seed, args.faster_config,
        ))?;

        // Phase 1: reservoir sample
        println!("Phase 1: reservoir sampling ({} clusters)...", nlist);
        {
            let f = std::fs::File::open(base)?;
            let all = read_fvecs_from_reader(BufReader::new(f), args.limit)?;
            for chunk in all.chunks(batch_size) {
                let mut flat = Vec::with_capacity(chunk.len() * dim);
                for v in chunk { flat.extend_from_slice(v); }
                builder.insert_batch(flat)?;
            }
        }
        // Phase 2: build
        println!("Phase 2: building...");
        let f2 = std::fs::File::open(base)?;
        let all2 = read_fvecs_from_reader(BufReader::new(f2), args.limit)?;
        let mut batch_iter = all2.chunks(batch_size).map(|c| {
            let mut flat = Vec::with_capacity(c.len() * dim);
            for v in c { flat.extend_from_slice(v); }
            flat
        }).collect::<Vec<_>>().into_iter();
        drop(all2);
        builder.build(Some(&mut || batch_iter.next()))?
    } else if let (Some(cp), Some(ap)) = (&args.centroids, &args.assignments) {
        let centroids = read_fvecs(cp, None)?;
        let assignments = read_ids(ap, None)?;
        IvfRabitqIndex::train_with_clusters(&vectors, &centroids, &assignments,
            args.bits, args.metric, RotatorType::FhtKacRotator, args.seed, args.faster_config)?
    } else {
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
        SaveFormat::V4 => v4_save_index(&index, save_path)?,
    }
    println!("  Saved in {:.1}s", t0.elapsed().as_secs_f64());
    println!("Done. Total: {:.1}s", total_start.elapsed().as_secs_f64());
    Ok(())
}
