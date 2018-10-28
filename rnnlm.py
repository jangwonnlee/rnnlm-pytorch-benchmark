
# coding: utf-8

# In[9]:


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[10]:


MASK_token = 0
EOS_token = 1

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "MASK", 1: "EOS"}
        self.n_words = 2  # Count MASK and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            #print(word)
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[11]:


print("Reading lines...")

# Read the file and split into lines
lines = open('./simple-examples/data/ptb.train.txt', encoding='utf-8').read().strip().split('\n')
print(len(lines))
#print(lines[0])

lang = Lang()

for line in lines:
    lang.addSentence(line.strip())
    
print("Counted words:")
print(lang.n_words)
#print(lang.index2word)

print(random.choice(lines))


# In[12]:


def read(lines, lang):
    for line in lines:
        sent = [lang.word2index[x] for x in line.strip().split()]
        sent.append(EOS_token)
        yield torch.LongTensor(sent)

train = list(read( lines, lang ))

lines_val = open('./simple-examples/data/ptb.valid.txt', encoding='utf-8').read().strip().split('\n')
#print(len(lines_val))

valid = list(read( lines_val, lang ))

MB_SIZE = 20

#print(train[:2])

train.sort(key=lambda x: -len(x))
train_order = list(range(0, len(train), MB_SIZE))

valid.sort(key=lambda x: -len(x))
valid_order = list(range(0, len(valid), MB_SIZE))

#print(len(valid))


# In[13]:


class RNNLM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNNLM, self).__init__()
        
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sequences):
        rnn_output, _ = self.lstm(self.embedding(sequences))
        #return self.softmax ( self.out(rnn_output.view(-1, self.hidden_size)) )
        return self.out(rnn_output.view(-1, self.hidden_size))


# In[14]:


def get_batch(sequences, volatile=False):
    lengths = torch.LongTensor([len(s) for s in sequences])
    batch = torch.LongTensor(lengths.max(), len(sequences)).fill_(MASK_token)
    for i, s in enumerate(sequences):
        batch[:len(s), i] = s
    batch = batch.to(device)
    return Variable(batch, volatile=volatile), lengths


# In[15]:


def train_step(batch, rnnlm, rnnlm_optimizer, criterion):
    
    #hidden = rnnlm.initHidden()
    rnnlm_optimizer.zero_grad()
    #length = tensor.size(0)
    #input = tensor[0]
    
    scores = rnnlm(batch[:-1])
    loss = criterion(scores, batch[1:].view(-1))

    loss.backward()
    rnnlm_optimizer.step()

    return loss.item()


# In[16]:


hidden_size = 256
rnnlm = RNNLM(hidden_size, lang.n_words).to(device)

print_every=1000
learning_rate=0.01

rnnlm_optimizer = optim.SGD(rnnlm.parameters(), lr=learning_rate)

weight = torch.FloatTensor(lang.n_words).fill_(1).to(device)
weight[MASK_token] = 0
criterion = nn.CrossEntropyLoss(weight, size_average=False)

i = 0
current_words = 0
print_loss_total = 0  # Reset every print_every

for ITER in range(13):
    random.shuffle(train_order)
    for sid in train_order:
        i += 1
        # train
        batch, lengths = get_batch(train[sid:sid + MB_SIZE])
        
        loss = train_step(batch, rnnlm, rnnlm_optimizer, criterion)
    
        print_loss_total += loss
        
        current_words += lengths.sum() - lengths.size(0)

        if i % print_every == 0:
            print_loss_avg = print_loss_total / current_words.item()
            print_loss_total = 0
            current_words = 0
            print('%.4f' % (print_loss_avg))
            
    print("epoch %r finished" % ITER)
    
    # log valid perplexity
    dev_loss = dev_words = 0
    for j in valid_order:
        batch, lengths = get_batch(valid[j:j + MB_SIZE])
        scores = rnnlm(batch[:-1])
        dev_loss += criterion(scores, batch[1:].view(-1)).item()
        dev_words += lengths.sum() - lengths.size(0)
    print("nll on valid = %.4f, ppl on valid = %.4f" % (dev_loss / dev_words.item(), np.exp(dev_loss / dev_words.item()) ))
    
    

