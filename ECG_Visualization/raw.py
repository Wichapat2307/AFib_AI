import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

AFDB_PATH = Path("mit-bih-atrial-fibrillation-database-1.0.0/files")
LTAF_PATH = Path("long-term-af-database-1.0.0/files")

DATASET = "ltaf"
RECORD_ID = "05"

WINDOW_SECONDS = 5


def load_record(dataset, record_id):

    db_path = AFDB_PATH if dataset == "afdb" else LTAF_PATH
    rec_path = str(db_path / record_id)

    record = wfdb.rdrecord(rec_path)

    print("Loaded:", record_id)
    print("Shape:", record.p_signal.shape)
    print("FS:", record.fs)

    return record


class RawViewer:

    def __init__(self, record):

        self.record = record
        self.signal = record.p_signal
        self.fs = int(record.fs)

        self.total_samples = len(self.signal)

        self.window_seconds = 5
        self.window_samples = int(self.window_seconds * self.fs)

        self.start = 0
        self.lead = 0

        self.fig, self.ax = plt.subplots(figsize=(16,5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update()
        plt.show()

    def update(self):

        self.ax.clear()

        end = min(self.start + self.window_samples, self.total_samples)

        segment = self.signal[self.start:end, self.lead]

        t = np.arange(self.start, end) / self.fs

        self.ax.plot(t, segment, linewidth=1)

        self.ax.set_title(
            f"Record {self.record.record_name} | "
            f"Lead {self.lead} | "
            f"{self.start/self.fs:.1f}s → "
            f"{end/self.fs:.1f}s | "
            f"Window: {self.window_seconds}s"
)

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        self.fig.canvas.draw_idle()

    def on_key(self, event):

        step = int(self.window_samples * 0.5)

        # move right
        if event.key == "right":
            self.start = min(
                self.start + step,
                self.total_samples - self.window_samples
            )

        # move left
        elif event.key == "left":
            self.start = max(self.start - step, 0)

        # zoom in (smaller window)
        elif event.key == "up":

            self.window_seconds = max(1, self.window_seconds - 1)
            self.window_samples = int(self.window_seconds * self.fs)

            # keep view centered
            self.start = max(0, self.start)

        # zoom out (larger window)
        elif event.key == "down":

            self.window_seconds += 1
            self.window_samples = int(self.window_seconds * self.fs)

        # lead switch
        elif event.key == "1":
            self.lead = 0

        elif event.key == "2":
            if self.signal.shape[1] > 1:
                self.lead = 1

        elif event.key == "q":
            plt.close(self.fig)
            return

        self.update()


if __name__ == "__main__":

    record = load_record(DATASET, RECORD_ID)
    viewer = RawViewer(record)