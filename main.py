from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from ui import UiMainWindow, UiClickableSplashScreen

def main():
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    splash = UiClickableSplashScreen()
    splash.show_message()
    window = UiMainWindow()
    window.adjust_ui(app)
    window.show()
    splash.finish(window)
    app.exec()


if __name__ == '__main__':
    main()

# from pathlib import Path

# def count_images(folder_path: Path):
#     counts = {}
#     for category in ['TP', 'TN', 'FP', 'FN']:
#         category_path = folder_path / category
#         if category_path.exists():
#             counts[category] = sum(file.is_file()
#                                for file in category_path.iterdir())
#         else:
#             counts[category] = 0
#     return counts

# def calculate_accuracy(counts):
#     TP = counts['TP']
#     TN = counts['TN']
#     FP = counts['FP']
#     FN = counts['FN']
#     total = TP + TN + FP + FN
#     return (TP + TN) / total if total else 0

# caffe_path = Path("H:\\Caffe")
# yunet_path = Path("H:\\YuNet")

# caffe_counts = count_images(caffe_path)
# yunet_counts = count_images(yunet_path)

# caffe_accuracy = calculate_accuracy(caffe_counts)
# yunet_accuracy = calculate_accuracy(yunet_counts)

# print(f"Caffe Accuracy: {caffe_accuracy:.2%}")
# print(f"YuNet Accuracy: {yunet_accuracy:.2%}")
