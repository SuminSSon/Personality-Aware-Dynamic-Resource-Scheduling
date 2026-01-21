import threading, time, psutil, torch, subprocess, re

class MetricsMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self.latest = {
            'cpu_usage': 0.0,
            'gpu_usage': 0.0,
            'memory_usage': 0.0,
            'cpu_temp': 0.0,
            'gpu_temp': 0.0,
            'gpu_mem_percent': 0.0,
            'cpu_cores': 0
        }
        # Initialize CPU cores
        self.latest['cpu_cores'] = psutil.cpu_count(logical=False)
        
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join()

    def _loop(self):
        while not self._stop.is_set():
            # CPU
            self.latest['cpu_usage'] = psutil.cpu_percent(interval=None)
            # GPU
            if torch.cuda.is_available():
                self.latest['gpu_usage'] = torch.cuda.memory_allocated() / 1e6
                total = torch.cuda.get_device_properties(0).total_memory / 1e6
                self.latest['gpu_mem_percent'] = (self.latest['gpu_usage'] / total) * 100
            # Memory
            self.latest['memory_usage'] = psutil.Process().memory_info().rss / 1e6
            # Temps
            try:
                out = subprocess.check_output(['sensors'], text=True)
                temps = re.findall(r'Core\s\d+:\s+\+(\d+\.\d+)', out)
                self.latest['cpu_temp'] = max(map(float, temps))
            except: pass
            try:
                out = subprocess.check_output(
                    ['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],
                    text=True
                )
                self.latest['gpu_temp'] = float(out.splitlines()[0])
            except: pass

            time.sleep(self.interval)