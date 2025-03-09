import collections.abc as c
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import batched
from typing import Optional, Union

from core.job import Job
from core.operation_types import FaceToolPair
from .cropper import Cropper


class BatchCropper(Cropper):
    """
    A class that manages image-cropping tasks using a thread pool.
    """

    def __init__(self, face_detection_tools: c.Iterator[FaceToolPair]):
        super().__init__()
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: list[Future] = []

        self.face_detection_tools = list(face_detection_tools)
    
    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}("
            f"threads={self.THREAD_NUMBER}, "
            f"progress_count={self.progress_count}, "
            f"end_task={self.end_task}, "
            f"show_message_box={self.show_message_box})>"
        )

    def terminate(self) -> None:
        """
        Terminates all pending tasks and shuts down the executor.
        """
        if not self.end_task:
            self.end_task = True
            self.emit_done()

        if self.executor:
            for future in self.futures:
                if not future.done():
                    future.cancel()
            self.executor.shutdown(wait=False)
            self.executor = None

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
    
    def all_tasks_done(self) -> None:
        """
        Checks if all futures have completed. If they have, shuts down the executor and emits done.
        This method should be called from the main thread.
        """
        if not self.end_task and all(future.done() for future in self.futures):
            if self.executor:
                self.executor.shutdown(wait=False)  # Non-blocking shutdown
            self.emit_done()
            self.end_task = True

    def worker_done_callback(self, future: Future) -> None:
        """
        Callback function to handle completion of a worker thread.
        """
        try:
            future.result()  # This raises any exceptions that occurred during execution
        except Exception as exc:
            self._display_error(
                exc, "An unexpected error occurred in a worker thread."
            )
        finally:
            # Check if all futures are done, then emit finished signal
            if all(f.done() for f in self.futures):
                self.all_tasks_done()
