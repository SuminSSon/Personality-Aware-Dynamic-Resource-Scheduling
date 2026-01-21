import numpy as np
from CustomCubatureKalmanFilter import CustomCubatureKalmanFilter
import psutil

class ResourceAllocator:
    def __init__(self,
                 min_batch_size:int, max_batch_size:int,
                 min_lr:float, max_lr:float,
                 gamma:float, target_accuracy:float,
                 ckf_params:dict, dyn_ratio_params:dict,
                 min_workers:int = 4,
                 max_workers:int = 8,
                 worker_k:float = 10.0,
                 worker_threshold:float = 0.5):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.target_accuracy = target_accuracy
        # CKF 초기화
        self.ckf = CustomCubatureKalmanFilter(
            ckf_params['state_dim'],
            ckf_params['obs_dim'],
            ckf_params['process_noise_cov'],
            ckf_params['measurement_noise_cov']
        )
        self.dyn_params = dyn_ratio_params
        self.allocation_weights = np.ones(15)
        self.allocation_bias = 0.0

        # --- num_workers용 파라미터 ---
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_k = worker_k
        self.worker_threshold = worker_threshold
        # cpu_free, mem_free → 2차원 입력
        self.worker_weights = np.random.randn(2)
        self.worker_bias = 0.0

        self.last_weights = None

    def preprocess_ckf(self, metrics:dict) -> (np.ndarray, np.ndarray):
        """
        metrics keys must be:
        ['ops','mu','gpu_usage','cpu_temp','gpu_temp','gpu_mem_percent','accuracy']
        Each value is a list of length num_nodes.
        Returns:
            x_filtered: np.ndarray shape (8, num_nodes)
            weights: np.ndarray shape (8, num_nodes)
        """
        # 메트릭 배열 생성
        arr = np.vstack([
            metrics['ops'],
            metrics['mu'],
            metrics['gpu_usage'],
            metrics['cpu_temp'],
            metrics['gpu_temp'],
            metrics['gpu_mem_percent'],  # corrected key name
            metrics['accuracy'],
            metrics['cpu_cores']
        ])  # shape: (8, N)
        # Min-Max 정규화
        mn = arr.min(axis=1, keepdims=True)
        mx = arr.max(axis=1, keepdims=True)
        norm = (arr - mn) / (mx - mn + 1e-5)

        # CKF 필터링
        x_filt = np.zeros_like(norm)
        for i in range(norm.shape[1]):
            self.ckf.predict()
            self.ckf.update(norm[:, i])
            x_filt[:, i] = self.ckf.state_estimate

        # 중요도 계산
        diff = np.abs(x_filt - 0.5)
        weights = diff / (diff.sum(axis=0, keepdims=True) + 1e-5)

        self.last_weights = weights 

        return x_filt, weights

    def allocate(self, node_index:int, metrics:dict, x_filtered:np.ndarray,
                 current_bs:int, current_lr:float) -> dict:
        """
        metrics keys must be same as preprocess.
        x_filtered: state estimates from CKF.
        Returns dict with 'shard_pct','batch_size','lr'.
        """
        # raw features
        raw = np.array([
            metrics['ops'][node_index],
            metrics['mu'][node_index],
            metrics['gpu_usage'][node_index],
            metrics['cpu_temp'][node_index],
            metrics['gpu_temp'][node_index],
            metrics['gpu_mem_percent'][node_index],
            metrics['accuracy'][node_index]
        ])
        # 피처 벡터 결합
        # combine CKF state (8) + raw (7) = 15-dim
        feat = np.concatenate([x_filtered[:, node_index], raw])
        eff = float(np.dot(feat, self.allocation_weights) + self.allocation_bias)
        eff_norm = eff / 100.0

        # 1) shard_pct (use_ratio)
        lower = self.dyn_params['lower']
        upper = self.dyn_params['upper']
        k = self.dyn_params['k']
        th = self.dyn_params['threshold']
        sig = 1 / (1 + np.exp(-k * (eff - th)))
        use_ratio = float(lower + (upper - lower) * sig)

        # 2) batch_size clamp
        bs = int(round(current_bs * (1 + self.gamma * eff_norm) / 2) * 2)
        bs = max(self.min_batch_size, min(bs, self.max_batch_size))

        # 3) learning_rate clamp
        acc_frac = metrics['accuracy'][node_index] / 100.0
        tgt_frac = self.target_accuracy / 100.0
        lr = float(current_lr * (1 + 0.1 * (tgt_frac - acc_frac)))
        lr = max(self.min_lr, min(lr, self.max_lr))

        # 추가: num_workers 계산
        # metrics 에 'cpu_usage' (%), 'proc_mem' (MB) 가 포함되어야 합니다.
        core_count = metrics['cpu_cores'][node_index]
        imp_core = self.last_weights[7, node_index]  # index 7 for cpu_cores
        num_workers = int(round(core_count * imp_core))
        num_workers = max(self.min_workers, min(num_workers, self.max_workers))

        '''
        cpu_free = 1.0 - metrics['cpu_usage'][node_index] / 100.0
        total_mem_mb = psutil.virtual_memory().total / (1024**2)
        
        proc_mem = metrics.get('proc_mem', metrics['memory_usage'][node_index])
        mem_free = max(0.0, 1.0 - proc_mem / total_mem_mb)

        w_feat = np.array([cpu_free, mem_free])
        w_eff  = float(self.worker_weights.dot(w_feat) + self.worker_bias)
        w_sig  = 1.0 / (1.0 + np.exp(-self.worker_k * (w_eff - self.worker_threshold)))
        num_workers = int(round(self.min_workers + (self.max_workers - self.min_workers) * w_sig))
        num_workers = max(self.min_workers, min(num_workers, self.max_workers))
        '''

        return {
            'shard_pct': use_ratio,
            'batch_size': bs,
            'lr': lr,
            'num_workers': num_workers
        }
