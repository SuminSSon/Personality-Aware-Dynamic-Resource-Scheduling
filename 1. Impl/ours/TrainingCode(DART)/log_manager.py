# log_manager.py
import csv, os
from threading import Lock

class LogManager:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self._lock = Lock()
        # 파일이 없으면 헤더부터 생성
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        # row: { key: value, ... } 로 fieldnames에 맞춰 넘기면 됨
        with self._lock, open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
