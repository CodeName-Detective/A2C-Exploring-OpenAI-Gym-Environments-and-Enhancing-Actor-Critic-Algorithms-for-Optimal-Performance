import torch


class CriticFunction(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(CriticFunction, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=64)
        self.linear2 = torch.nn.Linear(in_features=64, out_features=32)
        self.linear3 = torch.nn.Linear(in_features=32, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ActorFunction_Categorical(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorFunction_Categorical, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=64)
        self.linear2 = torch.nn.Linear(in_features=64, out_features=32)
        self.linear3 = torch.nn.Linear(in_features=32, out_features=output_size)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x

class ActorFunction_Continious(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorFunction_Continious, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=input_size, out_features=32)
        self.linear2 = torch.nn.Linear(in_features=32, out_features=16)
        self.mu_linear3 = torch.nn.Linear(in_features=16, out_features=output_size)
        self.sigma_linear3 = torch.nn.Linear(in_features=16, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        mu = torch.nn.functional.tanh(self.mu_linear3(x))
        sigma = torch.nn.functional.softplus(self.sigma_linear3(x))
        return mu, sigma