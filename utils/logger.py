import os
import sys
import time


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


class TxtLogger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def __call__(self, message):
        text = str(message)
        try:
            from tqdm import tqdm
            tqdm.write(text, file=sys.stderr)
        except Exception:
            print(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


class NullSummaryWriter(object):
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def build_summary_writer(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    try:
        from tensorboardX import SummaryWriter
        return SummaryWriter(log_dir)
    except Exception:
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir)
        except Exception:
            print("Warning: TensorBoard writer is unavailable.")
            return NullSummaryWriter()


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class PlainProgress(object):
    def __init__(self, iterable, total=None):
        self.iterable = iterable
        self.using_tqdm = False
        self.total = total
        self.n = 0

    def __iter__(self):
        return iter(self.iterable)

    def update(self, n=1):
        self.n += int(n)
        return None

    def set_postfix(self, *args, **kwargs):
        return None

    def close(self):
        return None


def build_progress_bar(iterable, enable=True, **kwargs):
    # Captured IDE logs often render tqdm carriage returns as many new lines.
    # Keep progress bars single-line only in real terminals; otherwise silence them.
    if not enable or not sys.stderr.isatty():
        return PlainProgress(iterable, total=kwargs.get("total"))
    try:
        from tqdm import tqdm
        kwargs.setdefault("file", sys.stderr)
        kwargs.setdefault("mininterval", 0.5)
        kwargs.setdefault("dynamic_ncols", True)
        bar = tqdm(iterable, **kwargs)
        bar.using_tqdm = True
        return bar
    except Exception:
        return PlainProgress(iterable, total=kwargs.get("total"))
