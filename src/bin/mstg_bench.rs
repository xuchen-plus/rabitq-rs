/// MSTG (Multi-Scale Tree Graph) index builder and benchmark.
///
/// Builds an MSTG index on GIST-1M, saves to disk, then benchmarks
/// with a recall-vs-QPS sweep over ef_search.
///
/// Usage:
///   # Build + benchmark
///   cargo run --release --bin mstg_bench -- \
///       --base data/gist/gist_base.fvecs \
///       --queries data/gist/gist_query.fvecs \
///       --gt data/gist/gist_groundtruth.ivecs \
///       --bits 7 --save data/gist/mstg_7bit
///
///   # Benchmark only (reload existing index)
///   cargo run --release --bin mstg_bench -- \
///       --load data/gist/mstg_7bit \
///       --queries data/gist/gist_query.fvecs \
///       --gt data/gist/gist_groundtruth.ivecs

use rabitq_rs::io::{read_fvecs, read_groundtruth};
use rabitq_rs::mstg::{MstgConfig, MstgIndex, ScalarPrecision, SearchParams};
use rabitq_rs::Metric;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    base: Option<PathBuf>,
    queries: Option<PathBuf>,
    gt: Option<PathBuf>,
    save: String,
    load: Option<String>,
    bits: usize,
    max_posting_size: usize,
    closure_epsilon: f32,
    max_replicas: usize,
    centroid_precision: String,
    topk: usize,
    seed: u64,
    limit: Option<usize>,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut base = None;
    let mut queries = None;
    let mut gt = None;
    let mut save = String::from("data/gist/mstg_7bit");
    let mut load = None;
    let mut bits = 7usize;
    let mut max_posting_size = 5000usize;
    let mut closure_epsilon = 0.02f32;
    let mut max_replicas = 8usize;
    let mut centroid_precision = String::from("bf16");
    let mut topk = 100usize;
    let mut seed = 42u64;
    let mut limit = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => {
                i += 1;
                base = Some(PathBuf::from(&args[i]));
            }
            "--queries" => {
                i += 1;
                queries = Some(PathBuf::from(&args[i]));
            }
            "--gt" => {
                i += 1;
                gt = Some(PathBuf::from(&args[i]));
            }
            "--save" => {
                i += 1;
                save = args[i].clone();
            }
            "--load" => {
                i += 1;
                load = Some(args[i].clone());
            }
            "--bits" => {
                i += 1;
                bits = args[i].parse()?;
            }
            "--max-posting-size" => {
                i += 1;
                max_posting_size = args[i].parse()?;
            }
            "--closure-epsilon" => {
                i += 1;
                closure_epsilon = args[i].parse()?;
            }
            "--max-replicas" => {
                i += 1;
                max_replicas = args[i].parse()?;
            }
            "--centroid-precision" => {
                i += 1;
                centroid_precision = args[i].clone();
            }
            "--topk" => {
                i += 1;
                topk = args[i].parse()?;
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

    Ok(Args {
        base,
        queries,
        gt,
        save,
        load,
        bits,
        max_posting_size,
        closure_epsilon,
        max_replicas,
        centroid_precision,
        topk,
        seed,
        limit,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Limit rayon to physical cores only
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", "16");
    }

    let args = parse_args()?;

    println!("=== RaBitQ MSTG Index Builder + Benchmark ===");
    println!("Bits:            {}", args.bits);
    println!("MaxPostingSize:  {}", args.max_posting_size);
    println!("ClosureEpsilon:  {}", args.closure_epsilon);
    println!("MaxReplicas:     {}", args.max_replicas);
    println!("Precision:       {}", args.centroid_precision);
    println!();

    // ----------------------------------------------------------------
    // Build or load
    // ----------------------------------------------------------------
    let index = if let Some(ref load_path) = args.load {
        println!("Loading MSTG index from {load_path}...");
        let t0 = Instant::now();
        let index = MstgIndex::load_from_path(load_path)?;
        let elapsed = t0.elapsed();

        let mem_mb =
            MstgIndex::estimate_memory_mb(&index.centroid_index, &index.posting_lists);
        println!(
            "  Loaded in {:.1}s  ({:.1} MB, {} centroids, {} posting lists)",
            elapsed.as_secs_f64(),
            mem_mb,
            index.centroid_index.len(),
            index.directory.len(),
        );
        index
    } else {
        let base = args
            .base
            .as_ref()
            .ok_or("--base <path> is required for build")?;

        // Load vectors
        println!("Loading vectors from {}...", base.display());
        let t0 = Instant::now();
        let vectors = read_fvecs(base, args.limit)?;
        let elapsed = t0.elapsed();
        if vectors.is_empty() {
            return Err("no vectors loaded".into());
        }
        let dim = vectors[0].len();
        println!(
            "  Loaded {} vectors in {:.1}s ({:.1} MB raw float data)",
            vectors.len(),
            elapsed.as_secs_f64(),
            vectors.len() * dim * 4 / (1024 * 1024)
        );
        println!("  Dimension: {dim}");
        println!();

        // Build MSTG config
        let precision = match args.centroid_precision.as_str() {
            "fp32" => ScalarPrecision::FP32,
            "bf16" => ScalarPrecision::BF16,
            "fp16" => ScalarPrecision::FP16,
            "int8" => ScalarPrecision::INT8,
            other => {
                eprintln!("unknown centroid precision: {other}");
                std::process::exit(1);
            }
        };

        let config = MstgConfig {
            max_posting_size: args.max_posting_size,
            branching_factor: 10,
            balance_weight: 1.0,
            closure_epsilon: args.closure_epsilon,
            max_replicas: args.max_replicas,
            rabitq_bits: args.bits,
            faster_config: false,
            metric: Metric::L2,
            hnsw_m: 32,
            hnsw_ef_construction: 200,
            centroid_precision: precision,
            default_ef_search: 150,
            pruning_epsilon: 0.6,
        };

        // Build index
        println!("Building MSTG index...");
        let total_start = Instant::now();
        let index = MstgIndex::build(&vectors, config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let build_elapsed = total_start.elapsed();

        let mem_mb =
            MstgIndex::estimate_memory_mb(&index.centroid_index, &index.posting_lists);
        println!();
        println!(
            "  Build time:      {:.1}s",
            build_elapsed.as_secs_f64()
        );
        println!("  Vectors:         {}", vectors.len());
        println!("  Centroids:       {}", index.centroid_index.len());
        println!("  Posting lists:   {}", index.directory.len());
        println!("  Memory:          {:.1} MB", mem_mb);
        println!();

        // Save index
        println!("Saving index to {}...", args.save);
        let t0 = Instant::now();
        index.save_to_path(&args.save)?;
        println!("  Saved in {:.1}s", t0.elapsed().as_secs_f64());
        println!();

        index
    };

    // ----------------------------------------------------------------
    // Benchmark
    // ----------------------------------------------------------------
    let queries_path = args
        .queries
        .as_ref()
        .ok_or("--queries <path> is required")?;
    let gt_path = args.gt.as_ref().ok_or("--gt <path> is required")?;

    println!("--- Benchmark ---");
    println!("Loading queries from {}...", queries_path.display());
    let queries = read_fvecs(queries_path, None)?;
    println!("  {} queries, dim={}", queries.len(), queries[0].len());

    println!("Loading ground truth from {}...", gt_path.display());
    let gt = read_groundtruth(gt_path, None)?;
    println!("  {} ground truth rows", gt.len());

    assert_eq!(
        queries.len(),
        gt.len(),
        "query count {} != ground truth count {}",
        queries.len(),
        gt.len()
    );

    // Determine effective topk from ground truth depth
    let gt_depth: Vec<usize> = gt.iter().map(|r| r.len().min(args.topk)).collect();
    let effective_topk = *gt_depth.iter().min().unwrap_or(&args.topk);
    if effective_topk < args.topk {
        println!(
            "  Note: min ground truth depth is {effective_topk}, using as effective topk"
        );
    }
    let topk = effective_topk;

    // Build ground truth sets
    let gt_sets: Vec<HashSet<usize>> = gt
        .iter()
        .map(|row| row.iter().take(topk).copied().collect())
        .collect();

    // Sweep ef_search
    let pruning_epsilon = 0.6;
    let ef_search_values: &[usize] = &[50, 100, 150, 200, 300, 400];

    println!();
    println!(
        "{:>10} | {:>10} | {:>12}",
        "ef_search", "QPS", format!("Recall@{topk}")
    );
    println!("{}", "-".repeat(10 + 3 + 10 + 3 + 12));

    for &ef_search in ef_search_values {
        let params = SearchParams::new(ef_search, pruning_epsilon, topk);

        let t0 = Instant::now();
        let mut total_recall = 0usize;

        for (qid, query) in queries.iter().enumerate() {
            let results = index.search(query, &params); // MSTG: no ? needed, returns Vec directly
            let gt_set = &gt_sets[qid];
            let hits = results
                .iter()
                .filter(|r| gt_set.contains(&r.vector_id))
                .count();
            total_recall += hits;
        }

        let elapsed = t0.elapsed();
        let qps = queries.len() as f64 / elapsed.as_secs_f64();
        let recall = total_recall as f64 / (queries.len() * topk) as f64;

        println!(
            "{:>10} | {:>10.0} | {:>12.4}",
            ef_search, qps, recall,
        );

        // Early stop if recall saturated
        if recall >= 0.9999 {
            break;
        }
    }

    println!();
    println!("Done.");
    Ok(())
}
