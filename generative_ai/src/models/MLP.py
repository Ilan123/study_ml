from torch import nn
from torch import Tensor
import copy

class MLP(nn.Module):
    def __init__(self,
        layers_dims: list,
        activation_layer: nn.Module = nn.ReLU(),
        bias: bool = True,
        device=None,
        dtype=None,
        flat:bool=False,
    ) -> None:
        """
            Multi-layer perceptron with customizable architecture.
            
            Args:
                layers_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
                activation_layer: Activation function to use between layers
                bias: Whether to include bias terms
        """
        super().__init__()
        self.flat = flat
        assert len(layers_dims) > 2
        layers = []
        
        # Hidden layers with activation
        for in_features, out_features in zip(layers_dims[:-2], layers_dims[1:-1]):
            layers.append(nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype))
            layers.append(copy.deepcopy(activation_layer))
        
        # Output layer (no activation)
        layers.append(nn.Linear(layers_dims[-2], layers_dims[-1], bias=bias, device=device, dtype=dtype))
            
        self.model = nn.Sequential(*layers)
        

    def forward(self, x) -> Tensor:
        if self.flat:
            x = x.flatten(1)
        return self.model(x)
        