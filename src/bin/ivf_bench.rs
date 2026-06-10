/// IVF+RaBitQ benchmark: recall vs QPS sweep over nprobe.
///
/// Usage:
///   cargo run --release --bin ivf_bench -- \
///       --index data/gist/index_ivf_4096_7bit.bin \
///       --queries data/gist/gist_query.fvecs \
///       --gt data/gist/gist_groundtruth.ivecs \
///       --topk 100
///
/// Outputs a table:
///   nprobe | QPS     | Recall@100
///   -------|---------|-----------
///   5      | 12500   | 0.452
///   ...

use std::sync::Arc;
use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use rabitq_rs::io::{read_fvecs, read_groundtruth};
use rabitq_rs::{IvfRabitqIndex, SearchParams};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
enum StoreType { Fs, S3 }

#[derive(Debug)]
struct BenchArgs {
    index: PathBuf,
    queries: PathBuf,
    gt: PathBuf,
    topk: usize,
    v4: bool,
    store_type: StoreType,
    s3_endpoint: String,
    s3_bucket: String,
    s3_access_key: String,
    s3_secret_key: String,
}

fn create_v4_store(args: &BenchArgs) -> Arc<dyn ObjectStore> {
    match args.store_type {
        StoreType::Fs => {
            let _ = std::fs::create_dir_all(&args.index);
            Arc::new(LocalFileSystem::new_with_prefix(&args.index).expect("LocalFileSystem"))
        }
        StoreType::S3 => {
            use object_store::aws::AmazonS3Builder;
            use object_store::prefix::PrefixStore;

            let prefix = args.index.to_string_lossy().to_string();
            let s3 = AmazonS3Builder::new()
                .with_endpoint(&args.s3_endpoint)
                .with_access_key_id(&args.s3_access_key)
                .with_secret_access_key(&args.s3_secret_key)
                .with_bucket_name(&args.s3_bucket)
                .with_allow_http(true)
                .build()
                .expect("Failed to create S3 store");
            Arc::new(PrefixStore::new(s3, prefix))
        }
    }
}

fn parse_args() -> Result<BenchArgs, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut index = None;
    let mut queries = None;
    let mut gt = None;
    let mut topk = 100usize;
    let mut v4 = false;
    let mut store_type = StoreType::Fs;
    let mut s3_endpoint = "http://localhost:9000".to_string();
    let mut s3_bucket = String::new();
    let mut s3_access_key = "minioadmin1".to_string();
    let mut s3_secret_key = "minioadmin1".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => { i += 1; index = Some(PathBuf::from(&args[i])); }
            "--queries" => { i += 1; queries = Some(PathBuf::from(&args[i])); }
            "--gt" => { i += 1; gt = Some(PathBuf::from(&args[i])); }
            "--topk" => { i += 1; topk = args[i].parse()?; }
            "--v4" => { v4 = true; }
            "--store" => { i += 1; store_type = match args[i].as_str() { "fs"=>StoreType::Fs, "s3"=>StoreType::S3, o=>{eprintln!("unknown store: {o}");std::process::exit(1);}}; }
            "--s3-endpoint" => { i += 1; s3_endpoint = args[i].clone(); }
            "--s3-bucket" => { i += 1; s3_bucket = args[i].clone(); }
            "--s3-access-key" => { i += 1; s3_access_key = args[i].clone(); }
            "--s3-secret-key" => { i += 1; s3_secret_key = args[i].clone(); }
            other => { eprintln!("unknown argument: {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    Ok(BenchArgs {
        index: index.ok_or("--index <path> is required")?,
        queries: queries.ok_or("--queries <path> is required")?,
        gt: gt.ok_or("--gt <path> is required")?,
        topk,
        v4,
        store_type,
        s3_endpoint,
        s3_bucket,
        s3_access_key,
        s3_secret_key,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Limit rayon to physical cores
    if std::env::var("RAYON_NUM_THREADS").is_err() {
        std::env::set_var("RAYON_NUM_THREADS", "16");
    }

    let args = parse_args()?;

    // ----------------------------------------------------------------
    // 1. Load index
    // ----------------------------------------------------------------
    println!("Loading index from {}...", args.index.display());
    let t0 = Instant::now();
    let index = if args.v4 {
        let store = create_v4_store(&args);
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(IvfRabitqIndex::load_from_v4(store))?
    } else {
        IvfRabitqIndex::load_from_path(&args.index)?
    };
    println!(
        "  Loaded in {:.1}s  ({:.1} MB, {} clusters, {} vectors)",
        t0.elapsed().as_secs_f64(),
        index.estimate_memory_mb(),
        index.cluster_count(),
        index.len(),
    );

    // ----------------------------------------------------------------
    // 2. Load queries and ground truth
    // ----------------------------------------------------------------
    println!(
        "Loading queries from {}...",
        args.queries.display()
    );
    let queries = read_fvecs(&args.queries, None)?;
    println!("  {} queries, dim={}", queries.len(), queries[0].len());

    println!(
        "Loading ground truth from {}...",
        args.gt.display()
    );
    let gt = read_groundtruth(&args.gt, None)?;
    println!("  {} ground truth rows", gt.len());

    assert_eq!(
        queries.len(),
        gt.len(),
        "query count {} != ground truth count {}",
        queries.len(),
        gt.len()
    );

    // Count how many ground truth rows have at least topk entries
    let gt_depth: Vec<usize> = gt.iter().map(|r| r.len().min(args.topk)).collect();
    let effective_topk = *gt_depth.iter().min().unwrap_or(&args.topk);
    if effective_topk < args.topk {
        println!(
            "  Note: min ground truth depth is {effective_topk}, using as effective topk"
        );
    }

    let topk = effective_topk;

    // Build ground truth sets for fast recall lookup
    let gt_sets: Vec<std::collections::HashSet<u64>> = gt
        .iter()
        .map(|row| row.iter().take(topk).map(|&x| x as u64).collect())
        .collect();

    // ----------------------------------------------------------------
    // 3. nprobe sweep
    // ----------------------------------------------------------------
    let nprobe_list: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];
    let max_nprobe = nprobe_list
        .iter()
        .copied()
        .filter(|&n| n <= index.cluster_count())
        .last()
        .unwrap_or(64);

    println!();
    println!(
        "{:>6} | {:>9} | {:>12} | {:>12}",
        "nprobe", "QPS", "Recall@{topk}", "AvgDist@1"
    );
    println!(
        "{}",
        "-".repeat(6 + 3 + 9 + 3 + 12 + 3 + 12)
    );

    for &nprobe in nprobe_list {
        if nprobe > max_nprobe {
            break;
        }

        let params = SearchParams::new(topk, nprobe);

        let t0 = Instant::now();
        let mut total_recall = 0usize;
        let mut total_dist1 = 0.0f64;

        for (qid, query) in queries.iter().enumerate() {
            let results = index.search(query, params)?;

            // Recall
            let gt_set = &gt_sets[qid];
            let hits = results
                .iter()
                .filter(|r| gt_set.contains(&r.id))
                .count();
            total_recall += hits;

            // Avg distance of top-1
            if let Some(r) = results.first() {
                total_dist1 += r.score as f64;
            }
        }

        let elapsed = t0.elapsed();
        let qps = queries.len() as f64 / elapsed.as_secs_f64();
        let recall = total_recall as f64 / (queries.len() * topk) as f64;
        let avg_dist1 = total_dist1 / queries.len() as f64;

        println!(
            "{:>6} | {:>8.0} | {:>11.4} | {:>12.6}",
            nprobe, qps, recall, avg_dist1,
        );

        // Early stop if recall saturated
        if recall >= 0.9999 {
            break;
        }
    }

    Ok(())
}
