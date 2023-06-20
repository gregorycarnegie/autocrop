import pandas as pd
from PyQt6 import QtCore
from typing import Optional

class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        if self._df is not None:
            return self._df.shape[0]

    def columnCount(self, parent=None):
        if self._df is not None:
            return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if index.isValid() and role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            try:
                if orientation == QtCore.Qt.Orientation.Horizontal:
                    return str(self._df.columns[section])
                if orientation == QtCore.Qt.Orientation.Vertical:
                    return str(self._df.index[section])
            except IndexError:
                return None
        return None
    