use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion::{
    arrow::array::RecordBatch,
    datasource::listing::{ListingOptions, ListingTable, ListingTableConfig},
    execution::SessionStateBuilder,
    prelude::{DataFrame, SessionConfig, SessionContext, col, lit},
};
use datafusion_datasource::ListingTableUrl;
use datafusion_datasource_parquet::ParquetFormat;
use geodatafusion_parquet::table::ParquetGeoTable;

async fn setup_df(fixture: &str) -> DataFrame {
    let cfg = SessionConfig::new().set_bool("datafusion.execution.parquet.pushdown_filters", true);
    let state = SessionStateBuilder::new().with_config(cfg).build();
    let ctx = SessionContext::new_with_state(state).enable_url_table();
    geodatafusion::register(&ctx);

    let path = format!("../../fixtures/{fixture}");
    let opts = ListingOptions::new(Arc::new(ParquetFormat::new()));
    let cfg =
        ListingTableConfig::new(ListingTableUrl::parse(path).unwrap()).with_listing_options(opts);
    let inner =
        Arc::new(ListingTable::try_new(cfg.infer_schema(&ctx.state()).await.unwrap()).unwrap());
    let _ = ctx
        .register_table("table", Arc::new(ParquetGeoTable::new(inner)))
        .unwrap();

    ctx.sql("SELECT * FROM 'table'").await.unwrap()
}

fn filter_df(df: DataFrame) -> DataFrame {
    let x_min = -104.778104;
    let y_min = 39.597333;
    let x_max = -104.750642;
    let y_max = 39.614196;
    let intersects = df.registry().udf("st_intersects").unwrap();
    let wkt = format!(
        "POLYGON (({x_min} {y_max}, {x_min} {y_min}, {x_max} {y_min}, {x_max} {y_max}, {x_min} {y_max}))"
    );
    let int = intersects.call(vec![lit(wkt), col("geometry")]);
    df.filter(int.is_true()).unwrap()
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
    let fixture = "parquet/overture.zstd.parquet";
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
