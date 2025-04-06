from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from ui import UiMainWindow

def main():
    app = QApplication([])
    apply_stylesheet(app, theme='dark_teal.xml')
    window = UiMainWindow()
    window.adjust_ui(app)
    window.show()
    app.aboutToQuit.connect(window.cleanup_workers)
    app.exec()


if __name__ == '__main__':
    main()
    # from core.processing import test_pipeline
    # from core.face_tools import create_tool_pair
    # from core import Job

    # face_tool = create_tool_pair()
    # images = [
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\1.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_768x1024_03684a9198c20c02bd4b7460b2d368a7.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_768x1024_18bcf123c321368f75032cf6ba53afc0.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_768x1024_f96956425c29095a84fcb1cd9ec2a6ee.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_828x814_5b2aaa2fbe0825527056bc7dd56e59c7.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_828x1252_96a112dde104307c2f6290c8e8015b71.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_969x1024_3383d8b50d8ded218591aea7dbf81518.jpg",
    #     "I:\\V\Vina Sky\\Pictures\\OnlyFans\\VinaSkyy\\VinaSkyOFLeak_898x1230_5fa6d36b5f1c6983bf69ccc9eb602507.jpg",
    # ]
    # job = Job(
    #     400, 400, True, False, True, 50, 62, 1000, 1, 1, 1, 1,
    #     (False, True, False, False, False, False)
    # )
    # for image in images:
    #     test_pipeline(image, job, face_tool)


