import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# =========================================================================
# 1. LAN MODEL DEFINITION (Fixed and complete)
# =========================================================================

# The original code had a typo: layer2 should use 'hidden_features' in __init__
# and layer1 should output 'hidden_features' for layer2 to take it as input.
class LAN_ResidualBlock(nn.Module):
    """
    Least Action Network (LAN) Residual Block.
    
    This block implements the fundamental residual connection (x_{k+1} = x_k + f_k), 
    where the residual term f_k is interpreted as the 'velocity' (dot{q}). 
    During training, the magnitude of f_k is penalized to enforce the Principle 
    of Least Action (C_Action = ||f_k||^2).
    
    The block returns both the next state (x_{k+1}) and the residual (f_k).
    
    :param in_features: The size of the input feature vector (must match output size).
    :type in_features: int
    :param hidden_features: The size of the hidden layer within the residual function f.
    :type hidden_features: int
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        # f(x) function: two linear layers
        # Layer 1 transforms the input feature size to the hidden feature size
        self.layer1 = nn.Linear(in_features, hidden_features)
        # ReLU activation (Typo fix: 'ReLu' -> 'ReLU')
        self.relu = nn.ReLU() 
        # Layer 2 transforms back from hidden feature size to the block's input size
        self.layer2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x_k):
        """
        Performs the forward pass through the residual block.
        
        :param x_k: The input feature map (Position q) at layer k.
        :type x_k: torch.Tensor
        :returns: A tuple containing the next feature map (x_{k+1}) and the residual (f_k).
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        # Calculate the residual function f(x_k, theta_k)
        f_k = self.layer1(x_k)
        f_k = self.relu(f_k)
        f_k = self.layer2(f_k)
        
        # x_{k+1} = x_k + f_k (Residual connection)
        x_k_plus_1 = x_k + f_k
        
        # Return both the next state and the residual for loss calculation
        return x_k_plus_1, f_k
    
class LAN(nn.Module):
    """
    The full Least Action Network (LAN) model.
    
    This model stacks multiple LAN_ResidualBlock instances and prepares the 
    network's output (logits) along with a list of all residual terms (f_k) 
    needed for the Action Cost calculation.
    
    :param input_size: The dimensionality of the raw input data (e.g., 784 for flattened MNIST).
    :type input_size: int
    :param num_blocks: The number of LAN_ResidualBlock layers in the model.
    :type num_blocks: int
    :param num_classes: The number of output classes for classification.
    :type num_classes: int
    """
    def __init__(self, input_size, num_blocks, num_classes):
        super().__init__()
        self.fc_in = nn.Linear(input_size, 128)
        # Stack of LAN Blocks (hidden_features is 128 here)
        self.blocks = nn.ModuleList([LAN_ResidualBlock(128, 128) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Performs the forward pass and collects all layer residuals.
        
        :param x: The input tensor (a batch of data).
        :type x: torch.Tensor
        :returns: A tuple containing the final output logits and a list of residual tensors.
        :rtype: tuple[torch.Tensor, list[torch.Tensor]]
        """
        residuals = []
        x = self.fc_in(x)
        for block in self.blocks:
            x, f_k = block(x)
            residuals.append(f_k)
        
        logits = self.fc_out(x)
        return logits, residuals

# =========================================================================
# 2. DATA SETUP (Required to run the code)
# =========================================================================

# --- Hyperparameters ---
LAMBDA = 0.01  
BATCH_SIZE = 32
input_dim = 784
num_classes = 10
num_blocks = 4
EPOCHS = 5 # Run for a few epochs to see the effect

# Generate dummy data for a runnable example
N_SAMPLES = 1000
X_train = torch.randn(N_SAMPLES, input_dim) # 1000 samples, 784 features
# Generate random labels (0 to 9)
Y_train = torch.randint(0, num_classes, (N_SAMPLES,))

# Create a DataLoader for batch processing
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Instantiate Model and Optimizer ---
model = LAN(input_dim, num_blocks, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================================================
# 3. TRAINING LOOP (The original, now wrapped in epochs and running)
# =========================================================================

print(f"Starting LAN training for {EPOCHS} epochs (Lambda={LAMBDA})")

for epoch in range(EPOCHS):
    total_loss_epoch = 0
    total_task_loss_epoch = 0
    total_action_cost_epoch = 0
    
    for i, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad() # Reset gradients

        # 1. Forward Pass
        logits, residuals = model(data)

        # 2. Calculate L_Task (Cross-Entropy Loss)
        L_Task = F.cross_entropy(logits, labels)

        # 3. Calculate C_Action (Least Action Cost)
        # C_Action = Sum over all layers k of ||f(x_k)||^2
        # Note: torch.sum(r**2) calculates the squared L2 norm for the whole batch
        C_Action = sum([torch.sum(r**2) for r in residuals]) / BATCH_SIZE

        # 4. Calculate the Total Loss J(theta)
        # J(theta) = L_Task + lambda * C_Action
        J_total = L_Task + LAMBDA * C_Action

        # 5. Backward Pass & 6. Update Parameters
        J_total.backward()
        optimizer.step()
        
        # Aggregate losses for printing
        total_loss_epoch += J_total.item()
        total_task_loss_epoch += L_Task.item()
        total_action_cost_epoch += C_Action.item()
    
    avg_loss = total_loss_epoch / len(train_loader)
    avg_task_loss = total_task_loss_epoch / len(train_loader)
    avg_action_cost = total_action_cost_epoch / len(train_loader)
    
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    print(f"AVG Total Loss: {avg_loss:.4f}")
    print(f"AVG Task Loss (CE): {avg_task_loss:.4f}")
    print(f"AVG Action Cost (||f||^2): {avg_action_cost:.4f}")