import numpy as np
import torch
from torch import nn

from sklearn.base import BaseEstimator

class FFN(nn.Module):
    def __init__(self, n_units: int, n_conditions: int, embedding_size: int = 8, hidden_size: int = 16, dropout = 0.):
        super(FFN, self).__init__()
        
        # Embedding layer for units
        self.unit_embedding = nn.Embedding(n_units, embedding_size)
        
        # Linear layer to process the concatenated input
        self.linear_in = nn.Linear(embedding_size + n_conditions, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, 1)

    def forward(self, X):
        unit_id, condition = X[:, 0][:, None], X[:, 1:]
        
        # check data types of tensors
        unit_id = unit_id.int()
        # condition = condition.float()
        
        # Obtain embedding for the given unit IDs
        unit_embedding = self.unit_embedding(unit_id)
        
        # Concatenate unit embedding and conditions along the last dimension
        concatenated_input = torch.cat((unit_embedding, condition.unsqueeze(1)), dim=-1)  # Corrected usage
        
        # Pass the concatenated input through the linear layer
        hidden = self.activation(self.linear_in(concatenated_input))
        hidden = self.dropout(hidden)
        response_time = self.linear_out(hidden)
        return response_time.squeeze(1)


class FFNRegressor(BaseEstimator):
    
    def __init__(
        self,
        module: nn.Module, 
        criterion = nn.MSELoss, 
        optimizer = torch.optim.Adam, 
        max_epochs: int = 10, 
        batch_size: int = 1024, 
        lr: float = 0.01,
        device = torch.device('cpu'),
        verbose=True,
        ):
        
        super(FFNRegressor, self).__init__()
        
        self.device = device
        self.module = module.to(self.device)
        self.criterion = criterion()
        self.optimizer = optimizer(self.module.parameters(), lr=lr)
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.dataloader = torch.utils.data.DataLoader
        self.dataset = torch.utils.data.Dataset
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = torch.tensor(X.values.astype(np.float32) if hasattr(X, 'values') else X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y.values.astype(np.float32) if hasattr(y, 'values') else y, dtype=torch.float32, device=self.device)
        
        if self.verbose:
            print('\nepoch\ttrain loss')
        
        self.module.train()
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            idx_shuffled = torch.randperm(X.shape[0])
            X, y = X[idx_shuffled], y[idx_shuffled]
            n_batch_repetitions = 0
            
            for i in range(0, len(X), self.batch_size):
                X_batch, y_batch = X[i:i+self.batch_size], y[i:i+self.batch_size]
                
                prediction = self.module(X_batch)
                
                loss = self.criterion(prediction, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batch_repetitions += 1
            
            if self.verbose:
                print(f"{epoch + 1}/{self.max_epochs}\t{epoch_loss/n_batch_repetitions:.8f}")
        self.module.eval()
        
    def predict(self, X: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.module(X).detach().cpu().numpy()
    
    def set_device(self, device: torch.device):
        self.device = device
        self.module = self.module.to(device)
        