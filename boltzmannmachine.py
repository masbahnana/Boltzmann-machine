#Máquinas de Boltzmann são modelos probabilísticos (ou geradores) não supervisionados, baseados em energia. Isso significa que elas associam uma energia para cada configuração das variáveis que se quer modelar.

import torch # para Deep Learning
import torch.autograd as autograd # para autodiferenciação
import torch.nn as nn # para montar redes neurais
import torch.nn.functional as F # funções do Torch
import torch.optim as optim # para otimização com GDE
import numpy as np # para
# carrega os dados MNIST
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=False)
data = np.random.permutation(data.train.images) # só precisamos das imagens aqui

class RBM(nn.Module):
    def __init__(self, nv=28*28, nh=512, cd_steps=1):
        super(RBM, self).__init__()
        # inicializa os parâmetros da MBR
        self.W = nn.Parameter(torch.randn(nv, nh) * 0.01)
        self.bv = nn.Parameter(torch.zeros(nv))
        self.bh = nn.Parameter(torch.zeros(nh))
        self.cd_steps = cd_steps # define a forma de Contrastive Divergence

    def bernoulli(self, p):
        # return F.relu(torch.sign(p - autograd.Variable(torch.rand(p.size()).cuda())))  
        return F.relu(torch.sign(p - autograd.Variable(torch.rand(p.size()))))     
        
    def energy(self, v):
        b_term = v.mv(self.bv)
        linear_tranform = F.linear(v, self.W.t(), self.bh)
        h_term = linear_tranform.exp().add(1).log().sum(1)
        return (-h_term -b_term).mean()

    def sample_h(self, v):
        ph_given_v = torch.sigmoid(F.linear(v, self.W.t(), self.bh))
        return self.bernoulli(ph_given_v)
    
    def sample_v(self, h):
        pv_given_h = torch.sigmoid(F.linear(h, self.W, self.bv))
        return self.bernoulli(pv_given_h)

    def forward(self, v):
        vk = v.clone() # inicializa vk
        # realiza k passos de amostragem de Gibbs
        for step in range(self.cd_steps): 
            hk = self.sample_h(vk)
            vk = self.sample_v(hk)
        
        return v, vk.detach()
    
rbm = RBM()
# rbm.cuda() # move os parâmetros da rede para a GPU
optimizer = optim.Adam(rbm.parameters(), 0.001)

batch_size = 64 # tamanho do mini-lote
epochs = 25 # qtd de épocas de treinamento
for epoch in range(epochs):
    losses = []
    # loop de treinamento
    for i in range(0, len(data)-batch_size, batch_size):
        # cria os mini-lotes
        x_batch = data[i:i+batch_size]
        x_batch = torch.from_numpy(x_batch).float()
        # x_batch = x_batch.cuda()
        x_batch = autograd.Variable(x_batch).bernoulli()

        optimizer.zero_grad() # zera os gradientes computados anteriormente
        v, vk = rbm(x_batch) # realiza o forward-pass (CD com amostragens de Gibbs)
        loss = rbm.energy(v) - rbm.energy(vk) # computa o custo
        losses.append(loss.data[0])

        loss.backward() # realiza o backward-pass
        optimizer.step() # atualiza os parâmetros
    
    print('Custo na época %d: ' % epoch, np.mean(losses))
    if epoch % 5 == 0 and epoch > 0: # a cada 5 épocas
        rbm.cd_steps += 2 # aumenta os as iterações em CD
        print('Alterando para CD%d...' % rbm.cd_steps)