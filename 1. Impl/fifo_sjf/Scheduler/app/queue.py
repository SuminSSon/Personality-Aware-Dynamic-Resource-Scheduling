from collections import deque

class JobQueue:
    def __init__(self):
        self.q = deque()
    def push(self, job_spec): self.q.append(job_spec)
    def pop(self): return self.q.popleft() if self.q else None
    def __len__(self): return len(self.q)

