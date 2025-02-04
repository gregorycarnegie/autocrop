import collections.abc as c
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from typing import Optional, Union

from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class BatchCropper(Cropper):
    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__()
        self.face_detection_tools = list(face_detection_tools)

    def emit_progress(self, amount: int):
        self.progress_count = 0
        self.progress.emit(self.progress_count, amount)
        self.started.emit()

    def set_futures(self, worker: c.Callable[..., None],
                    amount: int,
                    job: Job,
                    list_1: Union[batched, list],
                    list_2: Optional[list] = None):
        self.executor = ThreadPoolExecutor(max_workers=self.THREAD_NUMBER)

        if list_2:
            self.futures = [
                self.executor.submit(worker, amount, job, tool_pair, old=old_chunk, new=new_chunk)
                for old_chunk, new_chunk, tool_pair in zip(list_1, list_2, self.face_detection_tools)
            ]
        else:
            self.futures = [
                self.executor.submit(worker, amount, chunk, job, tool_pair)
                for chunk, tool_pair in zip(list_1, self.face_detection_tools)
            ]

    def complete_futures(self):
        # Attach a done callback to handle worker completion
        for future in self.futures:
            future.add_done_callback(self.worker_done_callback)
