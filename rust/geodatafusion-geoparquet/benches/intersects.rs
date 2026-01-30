use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion::{
    arrow::array::RecordBatch,
    execution::SessionStateBuilder,
    functions::core::expr_ext::FieldAccessor as _,
    prelude::{col, lit, DataFrame, ParquetReadOptions, SessionConfig, SessionContext},
};
use geodatafusion_geoparquet::file_format::GeoParquetFormatFactory;

async fn setup_df(fixture: &str) -> DataFrame {
    let cfg = SessionConfig::new().set_bool("datafusion.execution.parquet.pushdown_filters", true);
    let file_format = Arc::new(GeoParquetFormatFactory::default());
    let state = SessionStateBuilder::new()
        .with_config(cfg)
        .with_file_formats(vec![file_format])
        .build();
    let ctx = SessionContext::new_with_state(state).enable_url_table();
    geodatafusion::register(&ctx);

    ctx.read_parquet(
        format!("../../fixtures/{fixture}"),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap()
}

fn filter_df(df: DataFrame) -> DataFrame {
    let x_min = -104.778104;
    let y_min = 39.597333;
    let x_max = -104.750642;
    let y_max = 39.614196;
    let mut filtered_df = df
        .filter(col("geometry_bbox").field("xmax").gt_eq(lit(x_min)))
        .unwrap();
    filtered_df = filtered_df
        .filter(col("geometry_bbox").field("ymax").gt_eq(lit(y_min)))
        .unwrap();
    filtered_df = filtered_df
        .filter(col("geometry_bbox").field("xmin").lt_eq(lit(x_max)))
        .unwrap();
    filtered_df = filtered_df
        .filter(col("geometry_bbox").field("ymin").lt_eq(lit(y_max)))
        .unwrap();

    let intersects = filtered_df.registry().udf("st_intersects").unwrap();
    let wkt = format!("POLYGON (({x_min} {y_max}, {x_min} {y_min}, {x_max} {y_min}, {x_max} {y_max}, {x_min} {y_max}))");
    let int = intersects.call(vec![lit(wkt), col("geometry")]);
    filtered_df.filter(int.is_true()).unwrap()
}

async fn query(df: DataFrame, expected_count: usize) -> Vec<RecordBatch> {
    let results = df.collect().await.unwrap();
    let n_matched = results.iter().fold(0, |acc, rb| acc + rb.num_rows());
    assert_eq!(expected_count, n_matched);
    results
}

fn criterion_benchmark(c: &mut Criterion) {
    let ctx = SessionContext::new();
    geodatafusion::register(&ctx);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_time()
        .build()
        .unwrap();
    let fixture = "geoparquet/overture.zstd.parquet";
    let df = rt.block_on(setup_df(fixture));
    let expected_count = 2804;

    c.bench_with_input(BenchmarkId::new("intersects", fixture), &df, |b, df| {
        b.to_async(&rt).iter_batched_ref(
            || filter_df(df.clone()),
            |filtered_df| query(filtered_df.clone(), expected_count),
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
