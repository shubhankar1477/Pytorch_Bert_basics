import torch 
import torch.nn as nn
import torch.optim as optim


def make_batch():
    input_batch = []
    target_batch = []
    
    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]
        input_batch.append(input)
        target_batch.append(target)
    return input_batch , target_batch


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(1, n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_hidden, bias=False)
        self.V = nn.Linear(n_hidden, n_class, bias=False)  # New linear layer
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m)
        tanh = torch.tanh(self.d.squeeze(0) + self.H(X))
        output = self.b.unsqueeze(0) + self.V(self.W(X)) + self.U(tanh)
        return output        



if __name__ == "__main__":
    
    n_step = 2 # n of steps 
    n_hidden = 2 # hidden state
    m = 2 # Embedding size
    
    sentences = ["i like dog", "i love coffee", "i hate milk"]
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # number of Vocabulary
    
    
    model = NNLM()
    
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(),lr = 0.001)
    
    
    input_batch ,  target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    
    
    for epoch in range(1000):
        optimiser.zero_grad()
        output = model(input_batch)
        loss = criterion(output,target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimiser.step()
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])