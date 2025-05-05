from contextlib import suppress
from typing import Optional

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
             role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        match role:
            case Qt.ItemDataRole.DisplayRole:
                if index.isValid():
                    with suppress(IndexError):
                        # Retrieve the entire row as a tuple
                        row = self._df.row(index.row())
                        # Access the desired column by its integer index
                        value = row[index.column()]
                        return str(value)
                return None
            case _:
                return None

    def load_dataframe(self, section: int, orientation: Qt.Orientation) -> Optional[str]:
        with suppress(IndexError, AttributeError):
            match orientation:
                case Qt.Orientation.Horizontal:
                    return str(self._df.columns[section])
                case Qt.Orientation.Vertical:
                    return str(section)
            return None
        return None

    def headerData(self, section: int,
                   orientation: Qt.Orientation,
                   role: Qt.ItemDataRole = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        match role:
            case Qt.ItemDataRole.DisplayRole:
                return self.load_dataframe(section, orientation)
            case _:
                return None
