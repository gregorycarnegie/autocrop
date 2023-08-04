from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWidgets import QWidget


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent: Optional[QWidget]=None):
        super(DataFrameModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent: Optional[QModelIndex]=None):
        # if self._df is not None:
        return self._df.shape[0]

    def columnCount(self, parent: Optional[QModelIndex]=None):
        # if self._df is not None:
        return self._df.shape[1]

    def data(self, index: QModelIndex,
             role: int=Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section: int,
                   orientation: Qt.Orientation,
                   role: int=Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if role == Qt.ItemDataRole.DisplayRole:
            try:
                if orientation == Qt.Orientation.Horizontal:
                    return str(self._df.columns[section])
                if orientation == Qt.Orientation.Vertical:
                    return str(self._df.index[section])
            except IndexError:
                return None
        return None
    