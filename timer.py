import time

class Timer:
    
    def __init__(self, verbose=False):
        self.verbose=verbose
        self.durations = {}
        self.current_task = None

    @property
    def duration(self):
        return self.durations[None]

    def task(self, name):
        self.current_task = name
        return self

    def __enter__(self):
        self.start = time.time()
        if self.verbose:
            print(f"{self.current_task} : ") 
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        self.durations[self.current_task] = duration
        if self.verbose:
            print(f"{duration}") 