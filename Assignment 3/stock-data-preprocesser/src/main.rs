use polars::lazy::dsl::*;
use polars::prelude::*;
use std::fs::{self, File, metadata};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::error::Error;

const COLUMNS: [&str; 5] = ["Open", "High", "Low", "Close", "Volume"];
const WINDOWS: [(usize, &str); 4] = [(5, "Week"), (25, "Month"), (63, "Quarter"), (252, "Year")];

fn read_file(filepath: &Path) -> LazyFrame {
    // Attempt to read CSV data
    match LazyCsvReader::new(filepath).finish() {
        Ok(df) => df,
        Err(_) => {
            println!("Error: The file {:?} does not contain any columns.", filepath);
            LazyFrame::default() // Return an empty DataFrame if the file is empty or has issues
        }
    }
}

fn get_filepaths() -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let path = Path::new("../Stocks/");
    let mut filepaths = Vec::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && metadata(&path).unwrap().len() > 0 {
            filepaths.push(path);
        }
    }
    Ok(filepaths)
}

fn seasonal_features(df: &mut LazyFrame) {
    *df = df.clone().with_column(
            col("Date").str().to_date(
                StrptimeOptions {
                    format: Some("%Y-%m-%d".into()), // Adjust format as needed
                    ..Default::default()
                }
            )
        )
        .with_columns([
            col("Date").dt().ordinal_day().alias("DayOfYear"),
            col("Date").dt().day().alias("DayOfMonth"),
            col("Date").dt().weekday().alias("DayOfWeek"),
            col("Date").dt().week().alias("WeekNumber"),
            col("Date").dt().month().alias("Month"),
            col("Date").dt().year().alias("Year"),
            col("Date").dt().quarter().alias("Quarter"),
        ]);
}

fn lag_features(df: &mut LazyFrame, lags: &[i64]) {
    let mut new_columns = Vec::new();

    for &lag in lags {
        new_columns.push(
            col("Open")
                .shift(lit(lag))
                .fill_null_with_strategy(FillNullStrategy::Forward(None))
                .alias(&format!("Open_Lag_{}", lag)),
        );
        new_columns.push(
            col("High")
                .shift(lit(lag))
                .fill_null(lit(0))
                .alias(&format!("High_Lag_{}", lag)),
        );
        new_columns.push(
            col("Low")
                .shift(lit(lag))
                .fill_null(lit(0))
                .alias(&format!("Low_Lag_{}", lag)),
        );
        new_columns.push(
            col("Close")
                .shift(lit(lag))
                .fill_null(lit(0))
                .alias(&format!("Close_Lag_{}", lag)),
        );
        new_columns.push(
            col("Volume")
                .shift(lit(lag))
                .fill_null(lit(0))
                .alias(&format!("Volume_Lag_{}", lag)),
        );
    }

    *df = df.clone().with_columns(new_columns);
}

fn statistical_features(df: &mut LazyFrame) -> Result<(), Box<dyn Error>> {
    let mut new_columns = Vec::new();

    for column in COLUMNS {
        // Weekly, monthly, quarterly, and yearly means
        for (window, suffix) in WINDOWS {
            new_columns.push(col(column).shift(lit(1)).rolling_mean(RollingOptionsFixedWindow{ 
                window_size: window,
                min_periods: 1,
                weights: None,
                center: false,
                fn_params: None, 
            })
                .alias(&format!("{}_Mean_{}", column, suffix)));
        
        }

        // Absolute deviations
        for (window, suffix) in WINDOWS {
            let rolling_mean_expr = col(column).shift(lit(1)).rolling_mean(RollingOptionsFixedWindow{
                window_size: window,
                min_periods: 1,
                weights: None,
                center: false,
                fn_params: None, 
            });
            let abs_dev = (col(column).shift(lit(1)) - rolling_mean_expr.clone()).abs()
                .rolling_mean(RollingOptionsFixedWindow{
                    window_size: window, 
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                });

            new_columns.push(abs_dev.alias(&format!("{}_AbsDev_{}", column, suffix)));
        }

        // Standard deviations
        for (window, suffix) in WINDOWS {
            new_columns.push(col(column).shift(lit(1)).rolling_std(RollingOptionsFixedWindow {
                window_size: window,
                min_periods: 1,
                weights: None,
                center: false,
                fn_params: None,
            }).alias(&format!("{}_Std_{}", column, suffix)));
        }
    }

    *df = df.clone().with_columns(new_columns);
    Ok(())
}

fn extract_features(df: &mut LazyFrame) -> Result<(), Box<dyn Error>> {
    seasonal_features(df);
    let lags: Vec<i64> = (1..=25).collect();
    lag_features(df, &lags);
    statistical_features(df)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let filepaths = get_filepaths()?;

    for filepath in filepaths {
        let df = read_file(&filepath);
        let mut df = df;
        extract_features(&mut df)?;

        // Construct the output file path
        let filename = filepath.file_stem().unwrap().to_string_lossy();
        let output_path = format!("../processed_files/{}.csv", filename);
        println!("Saving preprocessed data to '{}'...", output_path);
        let mut output_file = File::create(output_path).unwrap();
        CsvWriter::new(&mut output_file)
            .include_bom(false)
            .include_header(true)
            .with_batch_size(NonZero::new(1024).unwrap())
            .finish(&mut df.collect()?).unwrap();
    }

    Ok(())
}

