#CustomCubatureKalmanFilter.py
import numpy as np

class CustomCubatureKalmanFilter:
    def __init__(self, state_dim, obs_dim, process_noise_cov, measurement_noise_cov):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.state_estimate = np.zeros(state_dim)
        self.covariance_estimate = np.eye(state_dim) * 0.1  # 초기 공분산을 줄여서 초기 상태 신뢰성을 높임

    def state_transition(self, state):
        # 상태 전이 함수: 자원의 시간에 따른 동적 변화 모델을 추가
        # 예를 들어, 자원 사용이 시간에 따라 감소하거나 증가하는 경향을 반영
        return 0.8 * state + 0.1 * np.random.randn(self.state_dim)  # 약간의 랜덤성을 추가

    def observation_function(self, state):
        # 관측 함수: 실제 시스템에서 관측 가능한 값으로 변환
        return state + 0.05 * np.random.randn(self.obs_dim)  # 관측 잡음 추가

    def cubature_points(self, mean, cov):
        n = mean.shape[0]
        # 공분산 행렬의 촐레스키 분해를 통해 큐바쳐 포인트 생성
        sqrt_cov = np.linalg.cholesky(cov)
        cubature_points = np.sqrt(n) * np.hstack((sqrt_cov, -sqrt_cov))
        return mean[:, None] + cubature_points

    def predict(self):
        cubature_pts = self.cubature_points(self.state_estimate, self.covariance_estimate)
        predicted_pts = np.array([self.state_transition(pt) for pt in cubature_pts.T]).T
        self.state_estimate = np.mean(predicted_pts, axis=1)
        # 공분산 갱신: 예측된 점들의 분산과 프로세스 노이즈를 고려
        self.covariance_estimate = np.cov(predicted_pts) + self.process_noise_cov

    def update(self, measurement):
        cubature_pts = self.cubature_points(self.state_estimate, self.covariance_estimate)
        predicted_measurements = np.array([self.observation_function(pt) for pt in cubature_pts.T]).T
        predicted_measurement = np.mean(predicted_measurements, axis=1)
        # 혁신 공분산 계산 (예측된 측정치의 분산 + 측정 노이즈)
        innovation_cov = np.cov(predicted_measurements) + self.measurement_noise_cov
        # 교차 공분산 계산 (큐바쳐 포인트와 예측된 측정치 간의 공분산)
        cross_cov = np.cov(cubature_pts, predicted_measurements)[0:self.state_dim, self.state_dim:]
        # 칼만 이득 계산
        kalman_gain = np.dot(cross_cov, np.linalg.inv(innovation_cov + 1e-5))  # 수치 안정성을 위해 작은 값을 더함
        # 상태 업데이트
        self.state_estimate += np.dot(kalman_gain, (measurement - predicted_measurement))
        # 공분산 업데이트
        self.covariance_estimate -= np.dot(kalman_gain, np.dot(innovation_cov, kalman_gain.T))
        # 공분산 행렬의 대칭성 보장 및 음수 값 방지
        self.covariance_estimate = (self.covariance_estimate + self.covariance_estimate.T) / 2
        self.covariance_estimate = np.maximum(self.covariance_estimate, 1e-5)

    def smooth(self, data):
        # 데이터에 대한 예측 및 업데이트 단계 반복
        for measurement in data:
            self.predict()
            self.update(measurement)
        return self.state_estimate