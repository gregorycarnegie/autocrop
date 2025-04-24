from typing import Optional, Protocol

import polars as pl
from pathlib import Path


class TableLoader(Protocol):
    """
    Strategy protocol for loading tabular data.
    """
    def __call__(
        self,
        file: Path,
    ) -> Optional[pl.DataFrame]:
        ...