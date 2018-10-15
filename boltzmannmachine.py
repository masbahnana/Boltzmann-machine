import torch 
import torch.autograd as autograd
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("tmp/", one_hot=False)
data = np.random.permutation(data.train.images) 

class RBM(nn.Module):
    def __init__(self, nv=28*28, nh=512, cd_steps=1):
        super(RBM, self).__init__()
        
        self.W = nn.Parameter(torch.randn(nv, nh) * 0.01)
        self.bv = nn.Parameter(torch.zeros(nv))
        self.bh = nn.Parameter(torch.zeros(nh))
        self.cd_steps = cd_steps 

    def bernoulli(self, p):
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
      
        for step in range(self.cd_steps): 
            hk = self.sample_h(vk)
            vk = self.sample_v(hk)
        
        return v, vk.detach()
    
rbm = RBM()

optimizer = optim.Adam(rbm.parameters(), 0.001)

batch_size = 64 
epochs = 25 
for epoch in range(epochs):
    losses = []
    
    for i in range(0, len(data)-batch_size, batch_size):
     
        x_batch = data[i:i+batch_size]
        x_batch = torch.from_numpy(x_batch).float()
        
        x_batch = autograd.Variable(x_batch).bernoulli()

        optimizer.zero_grad() 
        v, vk = rbm(x_batch)
        loss = rbm.energy(v) - rbm.energy(vk) 
        losses.append(loss.data[0])

        loss.backward() 
        optimizer.step() 
    
    print('Custo na época %d: ' % epoch, np.mean(losses))
    if epoch % 5 == 0 and epoch > 0: # a cada 5 épocas
        rbm.cd_steps += 2
        print('Alterando para CD%d...' % rbm.cd_steps)
