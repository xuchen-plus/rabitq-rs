/// Build IVF+RaBitQ index from GIST-1M dataset and save to disk.
///
/// Usage (V3 single-file):
///   cargo run --release --bin ivf_rabitq -- \
///       --base data/gist/gist_base.fvecs \
///       --nlist 4096 --bits 7 --faster-config \
///       --save data/gist/index_ivf_4096_7bit.bin
///
/// Usage (V4 manifest+segments, S3-compatible):
///   cargo run --release --bin ivf_rabitq -- \
///       --base data/gist/gist_base.fvecs \
///       --nlist 4096 --bits 7 --faster-config \
///       --format v4 --save data/gist/v4_index/
///
/// Insert new vectors into a V4 index:
///   cargo run --release --bin ivf_rabitq -- \
///       --load data/gist/v4_index/ --format v4 \
///       --insert data/gist/gist_learn.fvecs

use rabitq_rs::io::{read_fvecs, read_ids};
use rabitq_rs::{IvfRabitqIndex, Metric, RotatorType};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
enum SaveFormat {
    V3, // single .bin file
    V4, // manifest.bin + per-cluster .seg files
}

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
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut base = None;
    let mut centroids = None;
    let mut assignments = None;
    let mut nlist = None;
    let mut bits = 7usize;
    let mut save = None;
    let mut load = None;
    let mut insert = None;
    let mut format = SaveFormat::V3;
    let mut faster_config = false;
    let mut metric = Metric::L2;
    let mut seed = 42u64;
    let mut limit = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => {
                i += 1;
                base = Some(PathBuf::from(&args[i]));
            }
            "--centroids" => {
                i += 1;
                centroids = Some(PathBuf::from(&args[i]));
            }
            "--assignments" => {
                i += 1;
                assignments = Some(PathBuf::from(&args[i]));
            }
            "--nlist" => {
                i += 1;
                nlist = Some(args[i].parse()?);
            }
            "--bits" => {
                i += 1;
                bits = args[i].parse()?;
            }
            "--save" => {
                i += 1;
                save = Some(PathBuf::from(&args[i]));
            }
            "--load" => {
                i += 1;
                load = Some(PathBuf::from(&args[i]));
            }
            "--insert" => {
                i += 1;
                insert = Some(PathBuf::from(&args[i]));
            }
            "--format" => {
                i += 1;
                match args[i].as_str() {
                    "v3" => format = SaveFormat::V3,
                    "v4" => format = SaveFormat::V4,
                    other => {
                        eprintln!("unknown format: {other} (use v3 or v4)");
                        std::process::exit(1);
                    }
                }
            }
            "--faster-config" => {
                faster_config = true;
            }
            "--ip" => {
                metric = Metric::InnerProduct;
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            "--limit" => {
                i += 1;
                limit = Some(args[i].parse()?);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if load.is_none() {
        // Building new: need base + save
        if base.is_none() {
            return Err("--base <path> is required for building a new index".into());
        }
        if save.is_none() {
            return Err("--save <path> is required".into());
        }
        // Validate clustering params
        match (&centroids, &assignments, nlist) {
            (Some(_), Some(_), None) => {}
            (None, None, Some(_)) => {}
            (None, None, None) => {
                return Err(
                    "either --nlist or (--centroids + --assignments) is required".into(),
                );
            }
            _ => {}
        }
    }

    Ok(Args {
        base,
        centroids,
        assignments,
        nlist,
        bits,
        save,
        load,
        insert,
        format,
        faster_config,
        metric,
        seed,
        limit,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", "16");
    }

    let args = parse_args()?;

    // ── Load path: load existing index ──
    if let Some(ref load_path) = args.load {
        println!("=== RaBitQ IVF Index (load + insert) ===");
        println!("Load:     {}", load_path.display());
        println!("Format:   {:?}", args.format);
        println!();

        let t0 = Instant::now();
        let index = load_index(load_path, args.format)?;
        println!(
            "  Loaded in {:.1}s  ({} vectors, {} clusters, {:.1} MB)",
            t0.elapsed().as_secs_f64(),
            index.len(),
            index.cluster_count(),
            index.estimate_memory_mb(),
        );

        // Insert if requested
        if let Some(ref insert_path) = args.insert {
            println!("Inserting vectors from {}...", insert_path.display());
            perform_insert(index, insert_path, args.save.as_deref(), args.format)?;
        }

        return Ok(());
    }

    // ── Build path ──
    println!("=== RaBitQ IVF Index Builder ===");
    println!("Base:     {}", args.base.as_ref().unwrap().display());
    println!("Bits:     {}", args.bits);
    println!("Faster:   {}", args.faster_config);
    println!("Metric:   {:?}", args.metric);
    println!("Format:   {:?}", args.format);
    println!("Save to:  {}", args.save.as_ref().unwrap().display());
    println!();

    // 1. Load dataset
    let base = args.base.as_ref().unwrap();
    println!("Loading vectors from {}...", base.display());
    let t0 = Instant::now();
    let vectors = read_fvecs(base, args.limit)?;
    let elapsed = t0.elapsed();
    println!(
        "  Loaded {} vectors in {:.1}s ({:.1} MB)",
        vectors.len(),
        elapsed.as_secs_f64(),
        vectors.len() * vectors[0].len() * 4 / (1024 * 1024)
    );

    if vectors.is_empty() {
        return Err("no vectors loaded".into());
    }
    let dim = vectors[0].len();
    println!("  Dimension: {dim}");
    println!();

    // 2. Build index
    let total_start = Instant::now();

    let index = if let (Some(centroids_path), Some(assignments_path)) =
        (&args.centroids, &args.assignments)
    {
        println!(
            "Loading centroids from {}...",
            centroids_path.display()
        );
        let centroids = read_fvecs(centroids_path, None)?;
        println!("  Loaded {} centroids", centroids.len());
        if centroids[0].len() != dim {
            return Err(
                format!("centroid dimension {} != data dimension {dim}", centroids[0].len()).into(),
            );
        }
        println!(
            "Loading cluster assignments from {}...",
            assignments_path.display()
        );
        let assignments = read_ids(assignments_path, None)?;
        println!("  Loaded {} assignments", assignments.len());
        if assignments.len() != vectors.len() {
            return Err(format!(
                "assignment count {} != vector count {}",
                assignments.len(),
                vectors.len()
            )
            .into());
        }
        let nlist = centroids.len();
        println!(
            "\nBuilding IVF+RaBitQ index (pre-clustered, {nlist} clusters, {}-bit)...",
            args.bits
        );
        IvfRabitqIndex::train_with_clusters(
            &vectors,
            &centroids,
            &assignments,
            args.bits,
            args.metric,
            RotatorType::FhtKacRotator,
            args.seed,
            args.faster_config,
        )?
    } else {
        let nlist = args.nlist.ok_or("--nlist is required")?;
        println!(
            "Building IVF+RaBitQ index ({nlist} clusters, {}-bit, K-Means training)...",
            args.bits
        );
        IvfRabitqIndex::train(
            &vectors,
            nlist,
            args.bits,
            args.metric,
            RotatorType::FhtKacRotator,
            args.seed,
            args.faster_config,
        )?
    };

    let build_elapsed = total_start.elapsed();
    println!();
    println!("Index built in {:.1}s", build_elapsed.as_secs_f64());
    println!("  Vectors:     {}", index.len());
    println!("  Clusters:    {}", index.cluster_count());
    println!("  Memory:      {:.1} MB", index.estimate_memory_mb());
    println!();

    // 3. Save index
    let save_path = args.save.as_ref().unwrap();
    println!("Saving index to {}...", save_path.display());
    let t0 = Instant::now();
    save_index(&index, save_path, args.format)?;
    let elapsed = t0.elapsed();
    println!("  Saved in {:.1}s", elapsed.as_secs_f64());

    let total_elapsed = total_start.elapsed();
    println!("Done. Total time: {:.1}s", total_elapsed.as_secs_f64());

    Ok(())
}

// ── helpers ──

fn load_index(path: &Path, format: SaveFormat) -> Result<IvfRabitqIndex, Box<dyn std::error::Error>> {
    match format {
        SaveFormat::V3 => Ok(IvfRabitqIndex::load_from_path(path)?),
        SaveFormat::V4 => Ok(IvfRabitqIndex::load_from_v4_dir(path)?),
    }
}

fn save_index(
    index: &IvfRabitqIndex,
    path: &Path,
    format: SaveFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        SaveFormat::V3 => {
            index.save_to_path(path)?;
        }
        SaveFormat::V4 => {
            index.save_to_v4_dir(path)?;
        }
    }
    Ok(())
}

fn perform_insert(
    mut index: IvfRabitqIndex,
    insert_path: &Path,
    save_path: Option<&Path>,
    format: SaveFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let new_vectors = read_fvecs(insert_path, None)?;
    println!("  Loaded {} new vectors", new_vectors.len());

    if new_vectors.is_empty() {
        return Ok(());
    }

    let start_id = index.len();
    let t0 = Instant::now();

    index.batch_insert(start_id, &new_vectors)?;

    // Flush any remaining pending vectors before save
    index.flush_all_pending();

    let elapsed = t0.elapsed();
    println!(
        "  Inserted {} vectors in {:.1}s ({:.0} vec/s)",
        new_vectors.len(),
        elapsed.as_secs_f64(),
        new_vectors.len() as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  New total:      {} vectors, {:.1} MB",
        index.len(),
        index.estimate_memory_mb()
    );

    // Save if path provided
    if let Some(path) = save_path {
        println!("Saving to {}...", path.display());
        let t0 = Instant::now();
        save_index(&index, path, format)?;
        println!("  Saved in {:.1}s", t0.elapsed().as_secs_f64());
    }

    Ok(())
}
