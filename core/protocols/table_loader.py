from pathlib import Path
from typing import Protocol

import polars as pl


class TableLoader(Protocol):
    """
    Strategy protocol for loading tabular data.
    """
    def __call__(
        self,
        file: Path,
    ) -> pl.DataFrame:
        ...
