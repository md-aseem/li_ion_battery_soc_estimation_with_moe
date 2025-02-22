import torch

class SimpleEKF:
    def __init__(self, Q_max, Q=1e-5, R=0.01):
        self.Q_max = Q_max  # Battery capacity
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.soc = 1  # Initial SOC guess
        self.P = 0.1  # Initial error covariance

    def update(self, I_measured, V_measured, T_measured, dt, model):
        # Prediction step (Coulomb counting)
        self.soc_pred = self.soc + (I_measured * dt) / self.Q_max
        self.P_pred = self.P + self.Q

        # Compute NN Jacobian (dV/dSOC) using autograd
        soc_tensor = torch.FloatTensor([self.soc_pred]).requires_grad_(True)
        I_tensor = torch.FloatTensor([I_measured])
        T_tensor = torch.FloatTensor([T_measured])
        V_pred = model(torch.cat([soc_tensor, I_tensor, T_tensor]).unsqueeze(0))

        # Backprop to get gradient (Jacobian)
        V_pred.backward()
        H = soc_tensor.grad.item()  # dV/dSOC

        # Update step
        y = V_measured - V_pred.item()
        S = H * self.P_pred * H + self.R
        K = (self.P_pred * H) / S
        self.soc = self.soc_pred + K * y
        self.P = (1 - K * H) * self.P_pred

        return self.soc


class EnhancedEKF:
    def __init__(self, Q_max, sigma_i=0.1, R=0.01, sigma_ocv=0.01,
                 rest_threshold=0.05, rest_time=300):
        self.Q_max = Q_max
        self.sigma_i = sigma_i
        self.R = R
        self.sigma_ocv = sigma_ocv
        self.rest_threshold = rest_threshold
        self.rest_time = rest_time
        self.soc = None
        self.P = None
        self.last_current = None
        self.rest_duration = 0
        self.dt = None

    def initialize(self, V_measured, T_measured, model):
        soc_guess = torch.linspace(0, 1, 100, dtype=torch.float32)  # Ensure float32
        losses = []
        with torch.no_grad():
            for soc in soc_guess:
                inputs = torch.tensor([soc.item(), 0.0, T_measured], dtype=torch.float32).unsqueeze(0)  # Ensure float32
                V_pred = model(inputs)
                losses.append((V_pred.item() - V_measured) ** 2)
        best_idx = np.argmin(losses)
        self.soc = soc_guess[best_idx].item()
        soc_tensor = torch.tensor([self.soc], dtype=torch.float32, requires_grad=True)  # Ensure float32
        inputs = torch.cat([soc_tensor,
                            torch.tensor([0.0], dtype=torch.float32),
                            torch.tensor([T_measured], dtype=torch.float32)]).unsqueeze(0)  # Ensure float32
        V_pred = model(inputs)
        V_pred.backward()
        H = soc_tensor.grad.item()
        soc_error = self.sigma_ocv / abs(H) if H != 0 else 0.5
        self.P = soc_error ** 2

    def update_rest_state(self, I_measured, dt):
        if abs(I_measured) < self.rest_threshold:
            self.rest_duration += dt
        else:
            self.rest_duration = 0

    def update(self, I_measured, V_measured, T_measured, dt, model):
        if self.soc is None:
            if self.rest_duration >= self.rest_time:
                self.initialize(V_measured, T_measured, model)
            else:
                self.update_rest_state(I_measured, dt)
                return None

        # Prediction step (FIXED SIGN)
        self.dt = dt
        self.soc_pred = self.soc - (I_measured * dt) / self.Q_max  # Changed to -
        Q_current = (self.dt / self.Q_max) ** 2 * (self.sigma_i ** 2)
        self.P_pred = self.P + Q_current

        # Update step
        soc_tensor = torch.tensor([self.soc_pred], dtype=torch.float32, requires_grad=True)  # Ensure float32
        I_tensor = torch.tensor([I_measured], dtype=torch.float32)  # Ensure float32
        T_tensor = torch.tensor([T_measured], dtype=torch.float32)  # Ensure float32
        inputs = torch.cat([soc_tensor, I_tensor, T_tensor]).unsqueeze(0)  # Ensure float32
        V_pred = model(inputs)
        V_pred.backward()
        H = soc_tensor.grad.item()
        y = V_measured - V_pred.item()
        S = H * self.P_pred * H + self.R
        K = (self.P_pred * H) / S
        self.soc = self.soc_pred + K * y
        self.P = (1 - K * H) * self.P_pred
        self.update_rest_state(I_measured, dt)
        return self.soc


class AEKF:
    def __init__(self, Q_max, Q_init=1e-5, R_init=0.01, alpha=0.99, gamma=0.1):
        """
        Initialize the AEKF.

        Args:
            Q_max (float): Battery capacity.
            Q_init (float): Initial process noise covariance.
            R_init (float): Initial measurement noise covariance.
            alpha (float): Forgetting factor for noise adaptation (close to 1).
            gamma (float): Positive fitness coefficient for noise adaptation.
        """
        self.Q_max = Q_max  # Battery capacity
        self.Q = Q_init  # Initial process noise covariance
        self.R = R_init  # Initial measurement noise covariance
        self.soc = 1.0  # Initial SOC guess
        self.P = 0.1  # Initial error covariance
        self.alpha = alpha  # Forgetting factor
        self.gamma = gamma  # Fitness coefficient

    def update(self, I_measured, V_measured, T_measured, dt, model):
        """
        Update the AEKF with new measurements.

        Args:
            I_measured (float): Measured current.
            V_measured (float): Measured voltage.
            T_measured (float): Measured temperature.
            dt (float): Time step.
            model (torch.nn.Module): Neural network flat_feature_models for voltage prediction.

        Returns:
            float: Updated SOC estimate.
        """
        # Prediction step (Coulomb counting)
        self.soc_pred = self.soc + (I_measured * dt) / self.Q_max
        self.P_pred = self.P + self.Q

        # Compute NN Jacobian (dV/dSOC) using autograd
        soc_tensor = torch.FloatTensor([self.soc_pred]).requires_grad_(True)
        I_tensor = torch.FloatTensor([I_measured])
        T_tensor = torch.FloatTensor([T_measured])
        V_pred = model(torch.cat([soc_tensor, I_tensor, T_tensor]).unsqueeze(0))

        # Backprop to get gradient (Jacobian)
        V_pred.backward()
        H = soc_tensor.grad.item()  # dV/dSOC

        # Update step
        y = V_measured - V_pred.item()  # Measurement residual
        S = H * self.P_pred * H + self.R  # Residual covariance
        K = (self.P_pred * H) / S  # Kalman gain

        # Update state and error covariance
        self.soc = self.soc_pred + K * y
        self.P = (1 - K * H) * self.P_pred

        # Adaptive noise update
        self._update_noise_covariance(y, H)

        return self.soc

    def _update_noise_covariance(self, residual, H):
        """
        Update the process noise covariance Q and measurement noise covariance R.

        Args:
            residual (float): Measurement residual (y - y_pred).
            H (float): Jacobian of the measurement function.
        """
        # Compute the forgetting factor
        mu = abs(residual)  # New interest
        lambda_k = self.alpha + (1 - self.alpha) * np.exp(-self.gamma * mu)

        # Update measurement noise covariance R
        self.R = lambda_k * self.R + (1 - lambda_k) * (residual ** 2 - H * self.P * H)

        # Update process noise covariance Q
        q_update = (1 - lambda_k) * (K * residual ** 2 * K.T + self.P - self.P_pred)
        self.Q = lambda_k * self.Q + q_update

        # Ensure Q and R remain positive definite
        self.Q = max(self.Q, 1e-10)
        self.R = max(self.R, 1e-10)