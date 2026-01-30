use std::any::Any;
use std::cmp::{self, Ordering};
use std::collections::HashSet;
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
use futures::TryStreamExt as _;
use geo::{BoundingRect as _, Intersects as _, coord};
use geo_traits::to_geo::ToGeoGeometry as _;
use geoarrow_array::GeoArrowArrayAccessor as _;
use geoarrow_array::array::WkbArray;
use object_store::ObjectStore;
use object_store::path::Path;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::{ParquetRecordBatchStreamBuilder, ProjectionMask};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use rstar::primitives::GeomWithData;
use rstar::{RTree, RTreeObject as _};
use wkt::TryFromWkt;

/// A TableProvider wrapper around ListingTable that adds spatial row group pruning
pub struct GeoParquetNextTable {
    /// The inner ListingTable that handles the actual scanning
    inner: Arc<ListingTable>,
    index: RTree<PageBounds>, // TODO: would need to be Vec<T> for real usage
    flat_index: Vec<PageBounds>,
    use_flat: bool,
}

impl std::fmt::Debug for GeoParquetNextTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeoParquetNextTable")
            .field("inner", &self.inner)
            .finish()
    }
}

impl GeoParquetNextTable {
    /// Create a new GeoParquetTable wrapping an existing ListingTable
    pub async fn new(
        listing_table: Arc<ListingTable>,
        state: &dyn Session,
        use_flat: bool,
    ) -> Result<Self> {
        // Get the object store and file path for building the geo index
        let url = listing_table.table_paths()[0].get_url();
        let store = state.runtime_env().object_store_registry.get_store(url)?;

        // List files from the listing table
        let (scan_files, _) = listing_table
            .list_files_for_scan(state, &Vec::new(), None)
            .await?;

        // Get the first file path (index 0)
        let file_path = scan_files[0].iter().next().unwrap().path().clone();

        // Build the geo index
        println!("Building geo index");
        let (index, flat_index) = build_geo_index(store, file_path, "geometry").await;
        println!("Geo index completed. Indexed {} pages.", flat_index.len());

        Ok(Self {
            inner: listing_table,
            index,
            flat_index,
            use_flat,
        })
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

async fn build_geo_index(
    store: Arc<dyn ObjectStore>,
    path: Path,
    geo_column: &str,
) -> (RTree<PageBounds>, Vec<PageBounds>) {
    let mut reader = ParquetObjectReader::new(store, path)
        .with_preload_column_index(true)
        .with_preload_offset_index(true);
    let mut md_reader = ParquetMetaDataReader::new()
        .with_column_index_policy(PageIndexPolicy::Required)
        .with_page_index_policy(PageIndexPolicy::Required);
    md_reader.try_load_via_suffix(&mut reader).await.unwrap();
    md_reader.load_page_index(&mut reader).await.unwrap();
    let md = md_reader.finish().unwrap();
    let col_idx = md
        .row_group(0)
        .schema_descr()
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if c.name() == geo_column {
                Some(i)
            } else {
                None
            }
        })
        .next_back()
        .unwrap();

    let col_root = md
        .file_metadata()
        .schema_descr()
        .get_column_root_idx(col_idx);
    let col_mask = ProjectionMask::roots(md.file_metadata().schema_descr(), [col_root]);

    let mut bounding_rects = Vec::new();
    for (rg_idx, _) in md.row_groups().iter().enumerate() {
        let oi = &md.offset_index().unwrap()[rg_idx][col_idx];
        let mut num_read = 0;
        for (p_idx, page) in oi.page_locations.iter().enumerate() {
            let rows_in_page = if p_idx + 1 < oi.page_locations.len() {
                oi.page_locations[p_idx + 1].first_row_index - page.first_row_index
            } else {
                md.row_group(rg_idx).num_rows() - page.first_row_index
            };
            let rows_in_page = rows_in_page as usize;

            let selection = vec![
                RowSelector::skip(num_read),
                RowSelector::select(rows_in_page),
            ];
            num_read += rows_in_page;
            let rb_builder = ParquetRecordBatchStreamBuilder::new(reader.clone())
                .await
                .unwrap()
                .with_projection(col_mask.clone())
                .with_row_groups(vec![rg_idx])
                .with_row_selection(selection.into());
            let stream = rb_builder.build().unwrap();
            let batches = stream.try_collect::<Vec<_>>().await.unwrap();

            let mut page_bounds = OverallBounds::new();
            for batch in batches {
                let field = batch.schema_ref().field(0);
                let array = batch.column(0);
                let wkb_array: WkbArray = (array.as_ref(), field).try_into().unwrap();

                for g in wkb_array.iter() {
                    let geom = g.as_ref().unwrap().as_ref().unwrap().to_geometry();
                    page_bounds.push(&geom.bounding_rect().unwrap());
                }
            }

            let bounding_rect = page_bounds.inner.to_polygon();
            bounding_rects.push(PageBounds::new(
                bounding_rect,
                PageInfo {
                    row_group_id: rg_idx,
                    page_id: p_idx,
                },
            ));
        }
    }

    (RTree::bulk_load(bounding_rects.clone()), bounding_rects)
}

type PageBounds = GeomWithData<geo::Polygon, PageInfo>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PageInfo {
    row_group_id: usize,
    page_id: usize,
}

impl cmp::PartialOrd for PageInfo {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl cmp::Ord for PageInfo {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self == other {
            return Ordering::Equal;
        }

        if self.row_group_id < other.row_group_id {
            return Ordering::Less;
        }

        if self.row_group_id == other.row_group_id && self.page_id < other.page_id {
            return Ordering::Less;
        }

        Ordering::Greater
    }
}

#[derive(Debug)]
struct OverallBounds {
    inner: geo::Rect,
}

impl OverallBounds {
    const INIT_MAX: geo::Coord<f64> = coord! { x: f64::INFINITY, y: f64::INFINITY };
    const INIT_MIN: geo::Coord<f64> = coord! { x: f64::NEG_INFINITY, y: f64::NEG_INFINITY };

    fn new() -> Self {
        Self {
            inner: geo::Rect::new(Self::INIT_MAX, Self::INIT_MIN),
        }
    }

    fn push(&mut self, rect: &geo::Rect) {
        if Self::INIT_MAX == self.inner.max() && Self::INIT_MIN == self.inner.min() {
            self.inner.set_max(rect.max());
            self.inner.set_min(rect.min());
        }

        let max = coord! {
            x: self.inner.max().x.max(rect.max().x),
            y: self.inner.max().y.max(rect.max().y),
        };
        let min = coord! {
            x: self.inner.min().x.min(rect.max().x),
            y: self.inner.min().y.min(rect.max().y),
        };
        self.inner.set_max(max);
        self.inner.set_min(min);
    }
}

#[async_trait]
impl TableProvider for GeoParquetNextTable {
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
                let filt = &spatial_filters[0];
                let scan_pages = if self.use_flat {
                    self.flat_index
                        .iter()
                        .filter(|i| filt.query_bounds.intersects(i.geom()))
                        .map(|pb| pb.data.clone())
                        .collect()
                } else {
                    // TODO: in real operations we'd need to evaluate the RTree criteria for each filter
                    self.index
                        .locate_in_envelope_intersecting(&filt.query_bounds.to_polygon().envelope())
                        .map(|pb| pb.data.clone())
                        .collect::<HashSet<_>>()
                };
                let ap = geo_access_plan(&md, &scan_pages);
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
    let mut reader = ParquetObjectReader::new(store, path)
        .with_preload_column_index(true)
        .with_preload_offset_index(true);
    let mut md_reader = ParquetMetaDataReader::new()
        .with_column_index_policy(PageIndexPolicy::Required)
        .with_page_index_policy(PageIndexPolicy::Required);
    md_reader.try_load_via_suffix(&mut reader).await.unwrap();
    md_reader.load_page_index(&mut reader).await.unwrap();
    md_reader.finish().unwrap()
}

fn geo_access_plan(
    parquet_meta: &ParquetMetaData,
    scan_pages: &HashSet<PageInfo>,
) -> ParquetAccessPlan {
    let mut access_plan = ParquetAccessPlan::new_all(parquet_meta.num_row_groups());
    if scan_pages.is_empty() {
        return access_plan;
    }

    let oi = parquet_meta.offset_index().unwrap();
    let col_idx = 33; // "geometry" column for our test dataset
    for (rg_index, _) in parquet_meta.row_groups().iter().enumerate() {
        let oi = &oi[rg_index][col_idx];
        let mut row_selections = Vec::new();
        for (p_idx, page) in oi.page_locations.iter().enumerate() {
            let rows_in_page = if p_idx + 1 < oi.page_locations.len() {
                oi.page_locations[p_idx + 1].first_row_index - page.first_row_index
            } else {
                parquet_meta.row_group(rg_index).num_rows() - page.first_row_index
            };
            let rows_in_page = rows_in_page as usize;

            if scan_pages.contains(&PageInfo {
                row_group_id: rg_index,
                page_id: p_idx,
            }) {
                row_selections.push(RowSelector::select(rows_in_page));
            } else {
                row_selections.push(RowSelector::skip(rows_in_page))
            }
        }
        access_plan.scan_selection(rg_index, RowSelection::from(row_selections));
    }

    access_plan
}

fn _spatial_column_indexes(
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
                    if c.name() == f._geometry_column {
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
    _geometry_column: String,
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
                _geometry_column: col.name.clone(),
            });
        }

        None
    }
}
