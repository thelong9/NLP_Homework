import torch 
import torch.nn as nn 
import numpy as np
class SimpleNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, window_size: int):
        """ 
        Intialize the model 
        
        Input:
        vocab_size: int, input size of the model since each word is a one-hot vector. Yhis is alos the output size of the model
        embedding_dim: int, the dimension of the word embedding
        hidden_dim: int, the dimension of the hidden layer
        window_size: int, the size of the window to consider the context of the word
        """
        super(SimpleNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        
        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim=embedding_dim) 
        self.relu1 = nn.LeakyReLU(negative_slope=0.15)
        self.flaten = nn.Flatten()
        self.fc1 =nn.Linear(in_features=embedding_dim*window_size, out_features=hidden_dim)
        self.relu2 = nn.LeakyReLU(negative_slope=0.15)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.relu3 = nn.LeakyReLU(negative_slope=0.15)
        self.fc3= nn.Linear(in_features=hidden_dim, out_features= vocab_size)
        
    def forward(self, x):  # x: (batch_size, window_size), dtype = torch.int
        # print(x.shape)
        x = self.embedding(x)   # x: (batch_size, window_size, embedding_dim)
        x = self.relu1(x)   
        x = self.flaten(x)   # x: (batch_size, window_size*embedding_dim)
        x = self.fc1(x)     # x: (batch_size, hidden_dim)        
        x = self.relu2(x)   
        x = self.fc2(x)     # x: (batch_size, hidden_dim)
        x = self.relu3(x)
        x = self.fc3(x)     # x: (batch_size, vocab_size)
        # x = torch.softmax(x, dim=1)
        return x
    
    
def main():
    vocab_size = 20 
    window_size = 2
    embedding_dim = 5 
    hidden_dim = 5
    model = SimpleNN(vocab_size, embedding_dim, hidden_dim, window_size)
    _input = np.random.rand(1, window_size)
    _input = torch.tensor(_input, dtype=torch.long)
    with torch.no_grad():
        output = model(_input)
    print(output)
    
if __name__ == "__main__":
    main()