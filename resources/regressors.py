import torch
from torch import nn


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
        condition = condition.float()
        
        # Obtain embedding for the given unit IDs
        unit_embedding = self.unit_embedding(unit_id)
        
        # Concatenate unit embedding and conditions along the last dimension
        concatenated_input = torch.cat((unit_embedding, condition.unsqueeze(1)), dim=-1)  # Corrected usage
        
        # Pass the concatenated input through the linear layer
        hidden = self.activation(self.linear_in(concatenated_input))
        hidden = self.dropout(hidden)
        response_time = self.linear_out(hidden)
        return response_time.squeeze(1)
    
    # def fit(self, X: torch.Tensor, observation: torch.Tensor, max_epochs: int = 10, batch_size: int = 1024, verbose=False):
    #     # check data types of tensors
    #     observation = observation.float()
        
    #     self.train()
    #     optimizer = optim.Adam(self.parameters(), lr=0.01)
    #     for epoch in range(max_epochs):
    #         epoch_loss = 0
    #         idx_shuffled = torch.randperm(X.shape[0])
    #         X, observation = X, observation[idx_shuffled]
    #         n_batch_repetitions = 0
            
    #         for i in range(0, len(X), batch_size):
    #             X_batch, observation_batch = X[i:i+batch_size], observation[i:i+batch_size]
                
    #             prediction = self.__call__(X_batch)
                
    #             loss = self.loss_fn(prediction, observation_batch)
                
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
                
    #             epoch_loss += loss.item()
    #             n_batch_repetitions += 1
            
    #         if verbose:
    #             print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss/n_batch_repetitions:.8f}")        
        