import torch
from .abstractclassifier import AbstractClassifier
from .hypernn import HyperNN
from ..utils import numpy_to_torch

class HyperNNClassifier(AbstractClassifier):
    def __init__(self, nboxes=2, dim=2, tau=1, alpha=1, 
                 device='cuda', loss_fn=None, l1_reg=1e-3, l2_reg=1e-3,
                 training_epochs=100, learning_rate=1e-1, patience_early_stopping=50, 
                 alpha_tau_decay_step=1, prediction_threshold=0.5, 
                 verbose=False, verbosity=10, random_state=42,
                 reg_warmup_epochs=0, reg_normalize=False, reg_warmup_start=0.0):
        
        if loss_fn is None:
            loss_fn = torch.nn.BCELoss()
        
        if patience_early_stopping is None:
            patience_early_stopping = torch.inf
        
        self.nboxes = nboxes
        self.dim = dim
        self.device = device
        self.loss_fn = loss_fn
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.learning_rate = learning_rate
        self.tau = tau
        self.alpha = alpha
        self.training_epochs = training_epochs
        self.patience_early_stopping = patience_early_stopping
        self.alpha_tau_decay_step = alpha_tau_decay_step
        self.prediction_threshold = prediction_threshold
        self.verbose = verbose
        self.verbosity = verbosity
        self.random_state = random_state
        self.reg_warmup_epochs = reg_warmup_epochs
        self.reg_warmup_start = reg_warmup_start
        self.reg_normalize = reg_normalize
        self.epochs_run = 0  # épocas efetivamente executadas no último ajuste
        self.scheduler = None
        
        self.setup_classifier()

    def setup_classifier(self):
        self.model = HyperNN(
            nboxes=self.nboxes,
            dim = self.dim,
            tau=self.tau, 
            alpha=self.alpha,
            random_state=self.random_state)

        if self.device != "cpu":
            self.model.block = self.model.block.to(self.device)
        
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, 
                patience=30, min_lr=1e-5)
        
    def _reg_factor(self, lam, ramp):
        """Fator do peso de uma penalidade: vai de reg_warmup_start/lam (início) a 1 (fim do aquecimento)."""
        if lam <= 0:
            return 1.0
        f0 = min(1.0, self.reg_warmup_start / lam)
        return f0 + (1.0 - f0) * ramp

    def fit(self, X, y):
        self.dim = X.shape[1]
        if not hasattr(self, 'model') or self.model is None: 
            self.setup_classifier()
        X = numpy_to_torch(X, self.device)
        y = numpy_to_torch(y, self.device)
        best_loss = torch.inf
        self.model.train()
        self.epochs_run = 0
        for epoch in range(self.training_epochs):
            self.epochs_run = epoch + 1
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            z = self.model(X)
            
            # Compute loss (penalidades com aquecimento opcional)
            in_warmup = epoch < self.reg_warmup_epochs
            ramp = min(1.0, epoch / self.reg_warmup_epochs) if self.reg_warmup_epochs > 0 else 1.0
            w1 = self._reg_factor(self.l1_reg, ramp)
            w2 = self._reg_factor(self.l2_reg, ramp)
            reg_n = float(self.model.block.length.numel()) if self.reg_normalize else 1.0
            J = self.loss_fn(z, y)
            J += w1 * self.l1_reg * torch.norm(self.model.block.length, p=1) / reg_n
            J += w2 * self.l2_reg * torch.sum(torch.pow(self.model.block.length, 2)) / reg_n
            
            loss = J.detach().item()
            
            # Backward pass
            J.backward()  
            
            # Update weights
            self.optimizer.step()
            
            # Decay alpha and tau
            if self.alpha_tau_decay_step > 0 and ((epoch+1) % self.alpha_tau_decay_step) == 0:
                self.model.block.tau = max(self.model.block.tau * 0.95, 0.02)
                self.model.alpha = max(self.model.alpha * 0.95, 0.02)
                
            # Durante o aquecimento não há scheduler nem parada antecipada
            if in_warmup:
                continue

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step(loss)
                
            if loss < best_loss:
                best_loss = loss
                _patience = self.patience_early_stopping # Reset patience
            else:
                _patience -= 1 # Reduce patience
            
            if not _patience or loss == 0.:
                break
                
            # Print progress
            if self.verbose:
                if (((epoch+1) % self.verbosity) == 0) or (epoch == self.training_epochs-1):
                    print("Epoch: ", epoch+1)
                    print("Loss: ", loss)

    def predict(self, X):
        y_pred = []
        self.model.eval()
        with torch.inference_mode():
            X = numpy_to_torch(X, self.device)
            probs = self.model(X).detach()
            y_pred = (probs > self.prediction_threshold).long()

        return y_pred.cpu()

    def set_hyperparameters(self, hyperparameters:dict):
        return HyperNNClassifier(**hyperparameters)
    
    def get_num_boxes(self):
        return torch.sum(torch.abs(self.model.block.length) > 1e-3).item()/(self.dim)