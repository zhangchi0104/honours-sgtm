import pandas as pd
from rich.console import Console
from rich.table import Table, Column


def visualize_results(df: pd.DataFrame, n_items_per_row=5):
    console = Console()

    table = Table(Column("Topic", style="cyan"),
                  *[str(v + 1) for v in range(n_items_per_row)])
    for topic in df.columns:
        col = df[topic].sort_values(ascending=False)
        words = col.head(n_items_per_row).index.to_list()
        table.add_row(topic, *words)
    console.print(table)
