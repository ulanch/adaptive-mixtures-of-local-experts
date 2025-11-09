"""
Adaptive Mixture of Local Experts (Jacobs et al., 1991)
A modular supervised learning system where:
- Multiple expert networks each learn different subtasks
- A gating network learns to route inputs to appropriate experts
- Uses mixture-of-Gaussians objective for competitive specialization
Key features:
- Linear experts for interpretable decision surfaces
- Softmax gating with temperature control
- Responsibility-weighted backprop for localized learning
"""

import numpy as np


def _softmax(x, axis=-1):
    """Numerically stable softmax implementation"""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


class LinearExpert:
    """
    Single linear expert network that learns a linear decision surface.
    Each expert specializes in a subset of the input space by learning
    to minimize error only on cases assigned to it by the gating network.
    """

    def __init__(self, input_dim, output_dim, seed=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(input_dim)  # Xavier-style initialization
        self.W = self.rng.randn(input_dim, output_dim) * scale
        self.b = self.rng.randn(output_dim) * scale
        self.x = None  # cached for backward pass
        self.output = None  # cached for backward pass

    def forward(self, x):
        self.x = x
        self.output = x @ self.W + self.b  # simple linear transformation
        return self.output

    def backward(self, grad_output):
        # Standard linear layer gradients
        grad_W = self.x.T @ grad_output
        grad_b = np.sum(grad_output, axis=0)
        grad_x = grad_output @ self.W.T
        return grad_W, grad_b, grad_x

    def update(self, grad_W, grad_b, learning_rate):
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b


class GatingNetwork:
    """
    Gating network that learns to route inputs to appropriate experts.
    Outputs normalized mixing proportions p_i for each expert using softmax.
    Temperature parameter tau controls sharpness of expert selection.
    """

    def __init__(self, input_dim, num_experts, seed=None):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(input_dim)
        self.W = self.rng.randn(input_dim, num_experts) * scale
        self.b = np.zeros(num_experts)  # start with no bias
        self.x = None  # cached for backward pass
        self.z = None  # pre-softmax logits
        self.output = None  # mixing proportions p_i
        self.tau = 1.0  # temperature for softmax

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        if getattr(self, "tau", 1.0) != 1.0:
            z = z / float(self.tau)  # temperature scaling
        self.z = z
        self.output = _softmax(z, axis=1)  # normalize to mixing proportions
        return self.output

    def backward(self, grad_output):
        # Softmax backward: grad_z = softmax * (grad_out - dot(grad_out, softmax))
        s = self.output
        g = grad_output
        dot = np.sum(g * s, axis=1, keepdims=True)
        grad_z = s * (g - dot)
        tau = float(getattr(self, "tau", 1.0))
        if tau != 1.0:
            grad_z = grad_z / tau  # account for temperature in backward pass
        grad_W = self.x.T @ grad_z
        grad_b = np.sum(grad_z, axis=0)
        grad_x = grad_z @ self.W.T
        return grad_W, grad_b, grad_x

    def update(self, grad_W, grad_b, learning_rate):
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b


class MixtureOfExperts:
    """
    Full mixture of experts system (Jacobs et al., 1991).
    Combines multiple expert networks with a gating network that learns
    to assign responsibility for each training case to appropriate experts.
    
    Uses mixture-of-Gaussians objective (Eq 1.3 in paper):
        E = -log Σ_i p_i exp(-||d - o_i||²/(2σ²))
    
    This encourages competition: experts with lower error get higher responsibility.
    """

    def __init__(self, input_dim, output_dim, num_experts, seed=None, sigma=2.0, tau=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.sigma = float(sigma)  # std dev for Gaussian mixture model
        self.tau = float(tau)  # temperature for gating softmax
        self.rng = np.random.RandomState(seed)
        self.gating_network = GatingNetwork(input_dim, num_experts, seed=seed)
        self.gating_network.tau = self.tau
        self.experts = []
        for i in range(num_experts):
            expert_seed = None if seed is None else seed + i + 1
            self.experts.append(LinearExpert(input_dim, output_dim, seed=expert_seed))
        self.expert_outputs = None  # shape: (N, E, C)
        self.mixing_proportions = None  # p_i from gating network
        self.responsibilities = None  # posterior probabilities for each expert

    def forward(self, x):
        # Get mixing proportions from gating network
        p = self.gating_network.forward(x)
        self.mixing_proportions = p
        # Run all experts on the input
        N = x.shape[0]
        E = self.num_experts
        C = self.output_dim
        self.expert_outputs = np.zeros((N, E, C), dtype=np.float64)
        for i, expert in enumerate(self.experts):
            self.expert_outputs[:, i, :] = expert.forward(x)
        # Combine expert outputs weighted by mixing proportions
        mixture_output = np.sum(self.expert_outputs * p[:, :, None], axis=1)
        return mixture_output

    def compute_loss(self, x, target):
        """
        Compute mixture-of-Gaussians negative log-likelihood (Eq 1.3 in paper).
        Also computes responsibilities h_i = posterior probability that expert i
        generated the target, used for weighted backprop (Eq 1.5 in paper).
        """
        self.forward(x)
        y = target
        p = self.mixing_proportions
        # Compute squared error for each expert
        diff = y[:, None, :] - self.expert_outputs
        sq = np.sum(diff * diff, axis=2)
        # Mixture-of-Gaussians log probability: log Σ_i p_i exp(-||error||²/(2σ²))
        inv2s2 = 1.0 / (2.0 * self.sigma * self.sigma)
        log_weighted = np.log(p + 1e-12) - inv2s2 * sq
        # Numerically stable log-sum-exp
        m = np.max(log_weighted, axis=1, keepdims=True)
        exp_shift = np.exp(np.clip(log_weighted - m, -50, 50))
        denom = np.sum(exp_shift, axis=1, keepdims=True)
        log_mix_prob = (m + np.log(denom))[:, 0]
        # Compute responsibilities (posterior probs) for each expert
        self.responsibilities = exp_shift / denom
        loss = -np.mean(log_mix_prob)
        return loss

    def backward(self, x, target, learning_rate):
        """
        Backprop using responsibility-weighted gradients (Eq 1.5 in paper).
        Key insight: each expert's gradient is weighted by its posterior
        responsibility h_i, not just the prior mixing proportion p_i.
        This adapts experts faster early in training.
        """
        invs2 = 1.0 / (self.sigma * self.sigma)
        # Update each expert weighted by its responsibility
        for i, expert in enumerate(self.experts):
            grad_output = (self.responsibilities[:, i:i+1] *
                           (self.expert_outputs[:, i, :] - target)) * invs2
            grad_W, grad_b, _ = expert.backward(grad_output)
            expert.update(grad_W, grad_b, learning_rate)
        # Update gating network to adjust mixing proportions
        eps = 1e-12
        grad_mixing = -(self.responsibilities / (self.mixing_proportions + eps))
        grad_W, grad_b, _ = self.gating_network.backward(grad_mixing)
        self.gating_network.update(grad_W, grad_b, learning_rate)

    def train_step(self, x, target, learning_rate):
        loss = self.compute_loss(x, target)
        self.backward(x, target, learning_rate)
        return loss

    def predict(self, x):
        return self.forward(x)

    def predict_class(self, x):
        out = self.forward(x)
        return np.argmax(out, axis=1)

    def predict_class_combined(self, x):
        """Predict class using weighted average of all expert outputs"""
        p = self.gating_network.forward(x)
        N = x.shape[0]; E = self.num_experts; C = self.output_dim
        expert_out = np.zeros((N, E, C), dtype=np.float64)
        for i, expert in enumerate(self.experts):
            expert_out[:, i, :] = expert.forward(x)
        combined = np.sum(expert_out * p[:, :, None], axis=1)
        return np.argmax(combined, axis=1)

    def get_expert_responsibilities(self):
        return self.responsibilities

    def get_mixing_proportions(self):
        return self.mixing_proportions

    def compute_accuracy(self, x, target, combined=True):
        y_true = np.argmax(target, axis=1)
        y_pred = self.predict_class_combined(x) if combined else self.predict_class(x)
        return np.mean(y_pred == y_true)

    def compute_squared_error(self, x, target):
        out = self.forward(x)
        return np.mean((out - target) ** 2.0)
