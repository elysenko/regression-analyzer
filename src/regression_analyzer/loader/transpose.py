import polars as pl


def transpose_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Transpose DataFrame, converting first column to headers."""
    new_headers = df.select(df.columns[0]).to_series().to_list()
    new_headers = [str(h) for h in new_headers]

    data_cols = df.columns[1:]

    transposed_data = {}
    transposed_data["index"] = data_cols

    for i, header in enumerate(new_headers):
        unique_header = header
        counter = 1
        while unique_header in transposed_data:
            unique_header = f"{header}_{counter}"
            counter += 1

        transposed_data[unique_header] = [df[col][i] for col in data_cols]

    return pl.DataFrame(transposed_data)


def should_transpose_heuristic(df: pl.DataFrame) -> bool:
    """Simple heuristic to guess if transpose is needed."""
    if df.height > df.width * 10:
        return False

    first_col = df.select(df.columns[0]).to_series()

    if first_col.dtype == pl.Utf8:
        unique_ratio = first_col.n_unique() / len(first_col)
        avg_len = first_col.str.len_chars().mean()

        if unique_ratio > 0.9 and avg_len < 30:
            return True

    return False
