from typing import Optional

import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QWidget


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent: Optional[QWidget] = None):
        super(DataFrameModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:
        return self._df.shape[0]

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:
        return self._df.shape[1]

    def data(self, index: QModelIndex,
             role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def load_dataframe(self, section: int, orientation: Qt.Orientation) -> Optional[str]:
        try:
            match orientation:
                case Qt.Orientation.Horizontal:
                    return str(self._df.columns[section])
                case Qt.Orientation.Vertical:
                    return str(self._df.index[section])
        except IndexError:
            return None

    def headerData(self, section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        match role:
            case Qt.ItemDataRole.DisplayRole:
                return self.load_dataframe(section, orientation)
            case _:
                return None
