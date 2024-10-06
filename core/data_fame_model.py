from typing import Optional

# import pandas as pd
import polars as pl
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QWidget


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pl.DataFrame, parent: Optional[QWidget] = None):
        super(DataFrameModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return self._df.shape[0]

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return self._df.shape[1]

    def data(self, index: QModelIndex,
             role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            try:
                # Retrieve the entire row as a tuple
                row = self._df.row(index.row())
                # Access the desired column by its integer index
                value = row[index.column()]
                return str(value)
            except IndexError:
                # Handle cases where row or column indices are out of bounds
                return None
        return None

    def load_dataframe(self, section: int, orientation: Qt.Orientation) -> Optional[str]:
        try:
            if orientation == Qt.Orientation.Horizontal:
                # Access column names directly
                return str(self._df.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(section)
        except IndexError:
            # Handle cases where the section index is out of bounds
            return None

    def headerData(self, section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        match role:
            case Qt.ItemDataRole.DisplayRole:
                return self.load_dataframe(section, orientation)
            case _:
                return None
