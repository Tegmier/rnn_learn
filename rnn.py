import torch
from torch import nn

# torch.nn.RNN(input_size, hidden_size, num_layers=1) 1: number of recurrent layers
# feature_len = 100, hidden_len = 20, num_layers = 1

print("------ Tegmier Routine Cuda check ------")
print(torch.cuda.is_available())
torch.cuda.set_device(0)
print(torch.cuda.current_device())

rnn = nn.RNN(100, 20, 1).cuda()

# seq_len = 10, batch =3, feature_len = 100
x = torch.rand(10, 100, 100).cuda()

# define the initial hidden layer state
h0 = torch.zeros(1, 3, 20).cuda()

# calculate the result
out, h = rnn(x, h0)

# torch.Size([10, 3, 20]) torch.Size([seq_len, batch_size, hidden_len])
print(out.shape)

# torch.Size([1, 3, 20]) torch.Size([num_layer, batch_size, hidden_len])
print(h.shape)