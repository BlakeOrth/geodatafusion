use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::{Constraints, Result, Statistics};
use datafusion::config::TableParquetOptions;
use datafusion::datasource::TableProvider;
use datafusion::datasource::listing::ListingTable;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::ParquetAccessPlan;
use datafusion_datasource_parquet::source::ParquetSource;
use geo::{BoundingRect as _, Intersects as _, coord};
use object_store::ObjectStore;
use object_store::path::Path;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use wkt::TryFromWkt;

/// A TableProvider wrapper around ListingTable that adds spatial row group pruning
#[derive(Debug)]
pub struct ParquetGeoTable {
    /// The inner ListingTable that handles the actual scanning
    inner: Arc<ListingTable>,
}

impl ParquetGeoTable {
    /// Create a new GeoParquetTable wrapping an existing ListingTable
    pub fn new(listing_table: Arc<ListingTable>) -> Self {
        Self {
            inner: listing_table,
        }
    }

    /// Extract spatial predicates from filter expressions
    fn extract_spatial_filters(&self, filters: &[Expr]) -> Vec<SpatialPredicate> {
        let mut spatial_filters = Vec::new();

        for filter in filters {
            if let Some(predicate) = SpatialPredicate::try_from_expr(filter) {
                spatial_filters.push(predicate);
            }
        }

        spatial_filters
    }
}

#[async_trait]
impl TableProvider for ParquetGeoTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.inner.constraints()
    }

    fn get_table_definition(&self) -> Option<&str> {
        self.inner.get_table_definition()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let spatial_filters = self.extract_spatial_filters(filters);
        if spatial_filters.is_empty() {
            return self.inner.scan(state, projection, filters, limit).await;
        }

        let url = self.inner.table_paths()[0].get_url();
        let source = Arc::new(
            ParquetSource::new(TableParquetOptions::default()).with_enable_page_index(true),
        );
        let mut builder = FileScanConfigBuilder::new(
            // TODO: This would need to work with more general sources
            ObjectStoreUrl::local_filesystem(),
            self.inner.schema(),
            source,
        );
        let (scan_files, _) = self
            .inner
            .list_files_for_scan(state, &Vec::new(), limit)
            .await?;
        let store = state.runtime_env().object_store_registry.get_store(url)?;
        for group in scan_files {
            for file in group.iter() {
                let mut file = file.clone();
                let md = read_parquet_meta(store.clone(), file.path().clone()).await;
                // read parquet geo meta
                // create access plan for each file
                let ap = geo_access_plan(&md, &spatial_filters);
                file = file.with_extensions(Arc::new(ap));
                builder = builder.with_file(file);
            }
        }

        Ok(DataSourceExec::from_data_source(builder.build()))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // TODO: implement more exact evaluation
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    fn statistics(&self) -> Option<Statistics> {
        self.inner.statistics()
    }
}

async fn read_parquet_meta(store: Arc<dyn ObjectStore>, path: Path) -> ParquetMetaData {
    let mut reader = ParquetObjectReader::new(store, path).with_preload_column_index(true);
    let mut md_reader = ParquetMetaDataReader::new();
    md_reader.try_load_via_suffix(&mut reader).await.unwrap();
    md_reader.finish().unwrap()
}

fn geo_access_plan(
    parquet_meta: &ParquetMetaData,
    spatial_filters: &[SpatialPredicate],
) -> ParquetAccessPlan {
    let mut access_plan = ParquetAccessPlan::new_all(parquet_meta.num_row_groups());
    let col_indexes = spatial_column_indexes(parquet_meta, spatial_filters);
    for (rg_index, rg) in parquet_meta.row_groups().iter().enumerate() {
        for col_i in col_indexes.iter() {
            let col = rg.column(*col_i);
            if let Some(geo_stats) = col.geo_statistics()
                && let Some(bbox) = geo_stats.bounding_box()
            {
                let bbox_rect = geo::Rect::new(
                    coord! { x: bbox.get_xmin(), y: bbox.get_ymin() },
                    coord! { x: bbox.get_xmax(), y: bbox.get_ymax() },
                );

                for filt in spatial_filters {
                    // TODO: we'd need to evaluate the actual filter criteria here if we were
                    // supporting filtering more generally
                    if !bbox_rect.intersects(&filt.query_bounds) {
                        access_plan.skip(rg_index)
                    }
                }
            }
        }
    }

    access_plan
}

fn spatial_column_indexes(
    parquet_meta: &ParquetMetaData,
    spatial_filters: &[SpatialPredicate],
) -> Vec<usize> {
    spatial_filters
        .iter()
        .flat_map(|f| {
            parquet_meta
                .row_group(0)
                .schema_descr()
                .columns()
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if c.name() == f.geometry_column {
                        Some(i)
                    } else {
                        None
                    }
                })
        })
        .collect()
}

/// Represents a spatial predicate extracted from a filter expression
#[derive(Debug, Clone)]
struct SpatialPredicate {
    query_bounds: geo::Rect,
    geometry_column: String,
}

impl SpatialPredicate {
    /// Try to extract a spatial predicate from an expression
    pub fn try_from_expr(expr: &Expr) -> Option<Self> {
        use datafusion::logical_expr::Expr;

        // Look for st_intersects(literal_wkt, col) pattern
        // TODO: we'd need to support all the various filter methods if we were trying to get
        // something more general here
        if let Expr::IsTrue(sf) = expr
            && let Expr::ScalarFunction(func) = sf.as_ref()
            && func.func.name() == "st_intersects"
            && func.args.len() == 2
            && let Expr::Literal(wkt_value, _) = &func.args[0]
            && let Expr::Column(col) = &func.args[1]
        {
            let extent = geo::Polygon::try_from_wkt_str(&wkt_value.to_string()).unwrap();

            return Some(Self {
                query_bounds: extent.bounding_rect().unwrap(),
                geometry_column: col.name.clone(),
            });
        }

        None
    }
}
