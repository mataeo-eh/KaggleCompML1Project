from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq 

def find_column_headers():
# 1. Set root directory
    data_dir = Path("/kaggle/input/MABe-mouse-behavior-detection")
    parquet_files = list(data_dir.rglob("*.parquet"))

# 2. Read only column names (schema) from each file
    file_columns = {}
    max_cols = 0

    for f in parquet_files:
    # Construct index: "grandparent/parent/filename"
        index_str = f.parent.parent.name + "/" + f.parent.name + "/" + f.name
        cols = pq.read_schema(f).names
        file_columns[index_str] = cols
        max_cols = max(max_cols, len(cols))

# 3. Create a DataFrame: rows = files, columns = enumerated column positions
    df_cols = pd.DataFrame(index=list(file_columns.keys()), columns=range(max_cols))

# 4. Fill DataFrame with column names
    for file_stem, cols in file_columns.items():
        df_cols.loc[file_stem, :len(cols)-1] = cols

# 5. Quick inspection
    print("Column consistency table (first 10 rows):")
    print(df_cols.head(10))
    print("Column consistency table (last 10 rows):")
    print(df_cols.tail(10))

# 6. Find files with missing or extra columns
    files_with_missing_cols = df_cols[df_cols.isnull().any(axis=1)]
    print("\nFiles with missing columns:")
    print(files_with_missing_cols)

# 7. List all unique column names across all files
    unique_columns = pd.unique(df_cols.values.ravel())
    unique_columns = [c for c in unique_columns if c is not None]
    print(f"\nUnique column names across all files ({len(unique_columns)}):")
    print(unique_columns)

# 8. Check which column positions have inconsistent names across files
    df_t = df_cols.T  # transpose
    mask = df_t.nunique() != 1
    inconsistent_columns = df_t.loc[:, mask]

    print("\nColumn positions with inconsistent names across files:")
    print(inconsistent_columns)

# Optional: show which files have inconsistencies in these columns
    files_with_inconsistencies = inconsistent_columns[inconsistent_columns.notnull().any(axis=1)]
    print("\nFiles with inconsistencies at these column positions:")
    print(files_with_inconsistencies)