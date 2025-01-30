Here's the formatted version ready for direct copy-paste into an .md file:

```markdown
# Pandas Functions Reference Guide

A comprehensive collection of Pandas functions with descriptions for data analysis and feature engineering.

## Table of Contents
1. [Data Import/Export Functions](#1-data-importexport-functions)
2. [Data Inspection Functions](#2-data-inspection-functions)
3. [Data Cleaning Functions](#3-data-cleaning-functions)
4. [Data Manipulation Functions](#4-data-manipulation-functions)
5. [Feature Engineering Functions](#5-feature-engineering-functions)
6. [Statistical Functions](#6-statistical-functions)
7. [Time Series Functions](#7-time-series-functions)
8. [Text Processing Functions](#8-text-processing-functions)
9. [Grouping Functions](#9-grouping-functions)
10. [Advanced Functions](#10-advanced-functions)

## 1. Data Import/Export Functions

### Reading Data
```python
# Read CSV file into DataFrame
pd.read_csv('file.csv', 
    sep=',',              # Delimiter to use
    header=0,             # Row number to use as column names
    encoding='utf-8',     # File encoding
    parse_dates=True)     # Parse date strings to datetime

# Read Excel file
pd.read_excel('file.xlsx',
    sheet_name='Sheet1',  # Specify sheet to read
    header=0,             # Row for column names
    index_col=None)       # Column to use as index

# Read SQL query or database table
pd.read_sql('SELECT * FROM table',
    connection,           # Database connection
    index_col='id')       # Column to set as index

# Read JSON formatted data
pd.read_json('file.json',
    orient='records',     # JSON string format
    lines=True)           # Read file as JSON lines
```

### Writing Data
```python
# Write DataFrame to CSV
df.to_csv('output.csv',
    index=False,          # Don't write index
    header=True,          # Write column names
    encoding='utf-8')     # File encoding

# Write to Excel file
df.to_excel('output.xlsx',
    sheet_name='Sheet1',  # Specify sheet name
    index=False)          # Don't write index

# Write records to SQL database
df.to_sql('table_name',
    connection,           # Database connection
    if_exists='replace')  # How to behave if table exists
```

## 2. Data Inspection Functions

### Basic Information
```python
# View first n rows
df.head(n=5)             # Default shows 5 rows

# View last n rows
df.tail(n=5)             # Default shows 5 rows

# Display DataFrame information
df.info(
    verbose=True,        # Print full summary
    memory_usage='deep') # Calculate full memory usage

# Generate descriptive statistics
df.describe(
    percentiles=[.05, .25, .75, .95], # Custom percentiles
    include='all')       # Include all dtypes
```

### Data Checking Functions
```python
# Check for missing values
df.isnull().sum()        # Count missing values per column

# Check for duplicate rows
df.duplicated(
    subset=['col1', 'col2'], # Check specific columns
    keep='first')            # Which duplicates to mark

# Count unique values
df['column'].value_counts(
    normalize=True,      # Return proportions
    dropna=False)        # Include NaN values
```

## 3. Data Cleaning Functions

### Missing Value Operations
```python
# Remove missing values
df.dropna(
    axis=0,              # 0 for rows, 1 for columns
    how='any',           # 'any' or 'all'
    subset=['col1'],     # Consider only specific columns
    inplace=False)       # Modify original DataFrame

# Fill missing values
df.fillna(
    value=0,             # Value to use for filling NaN
    method='ffill',      # Method: 'ffill', 'bfill'
    limit=None)          # Max number of consecutive fills

# Interpolate missing values
df.interpolate(
    method='linear',     # Interpolation method
    limit_direction='forward', # Direction to fill
    axis=0)              # 0 for rows, 1 for columns
```

## 4. Data Manipulation Functions

### Selection and Indexing
```python
# Label-based indexing
df.loc[
    row_indexer,         # Row labels/boolean array
    column_indexer]      # Column labels

# Integer-based indexing
df.iloc[
    row_indexer,         # Row positions
    column_indexer]      # Column positions

# Filter columns by label
df.filter(
    items=['col1', 'col2'], # List of columns to keep
    like='string',          # Keep columns containing string
    regex='pattern')        # Keep columns matching pattern
```

## 5. Feature Engineering Functions

### Mathematical Operations
```python
# Apply function to rows/columns
df.apply(
    func,                # Function to apply
    axis=0,             # 0 for columns, 1 for rows
    raw=False)          # Whether to pass ndarray

# Convert categorical to dummy variables
pd.get_dummies(
    df,                 # DataFrame or Series
    columns=['col1'],   # Columns to encode
    prefix=['prefix'])  # Prefix for dummy columns
```

## 6. Statistical Functions

### Basic Statistics
```python
# Calculate mean
df.mean(
    axis=0,             # 0 for columns, 1 for rows
    skipna=True)        # Skip NA/null values

# Calculate correlation
df.corr(
    method='pearson',   # Correlation method
    min_periods=1)      # Minimum number of observations
```

## 7. Time Series Functions

### Date Operations
```python
# Convert to datetime
pd.to_datetime(
    df['date'],         # Series to convert
    format='%Y-%m-%d',  # Date format
    errors='coerce')    # How to handle errors

# Resample time-series data
df.resample(
    rule='M',           # Resampling frequency
    on='date')          # Column to use as timestamp
```

## 8. Text Processing Functions

### String Operations
```python
# Check if pattern/regex is contained
df['text'].str.contains(
    pat='pattern',      # Pattern to search for
    case=True,          # Case sensitive
    regex=True)         # Use regex

# Extract matched patterns
df['text'].str.extract(
    pat='(\d+)',        # Pattern to extract
    expand=True)        # Return DataFrame
```

## 9. Grouping Functions

### Group Operations
```python
# Group DataFrame
df.groupby(
    by='column',        # Column(s) to group by
    as_index=False)     # Don't use group keys as index

# Aggregate using one or more operations
df.agg(
    func=['sum', 'mean'], # Functions to apply
    axis=0)               # 0 for columns, 1 for rows
```

## 10. Advanced Functions

### Complex Operations
```python
# Create pivot table
df.pivot_table(
    values='value',     # Column to aggregate
    index='row_fields', # Fields for rows
    columns='col_fields', # Fields for columns
    aggfunc='mean')     # Aggregation function

# Merge DataFrames
pd.merge(
    left=df1,           # Left DataFrame
    right=df2,          # Right DataFrame
    how='inner',        # Type of merge
    on='key')           # Columns to merge on
```

## Best Practices Example
```python
# Complete data processing pipeline
def process_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Create new features
    df['new_feature'] = df['col1'] / df['col2']
    
    # Group and aggregate
    result = df.groupby('category').agg({
        'col1': 'mean',
        'col2': 'sum',
        'col3': 'count'
    })
    
    return result
```

## Additional Resources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [API Reference](https://pandas.pydata.org/docs/reference/index.html)
- [User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

---
Note: This guide provides common usage patterns. For complete documentation and additional parameters, refer to the official Pandas documentation.
```

You can now directly copy this entire content and paste it into a .md file. The formatting will be preserved, including code blocks, headers, and links.