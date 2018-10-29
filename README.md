# rnnlm-pytorch-benchmark

This repository contains the rnnlm (recurrent neural network language model) training code implemented as pytorch.

In particular, this rnnlm training code is based on the PTB (Penn Tree Bank) data set, thus can be used for benchmark testing purposes. ([See Mikolov's RNNLM toolkit](http://www.fit.vutbr.cz/~imikolov/rnnlm/))

Training results were evaluated on a perplexity basis, with around 40 and 180 results respectively for training and validation data sets.

To run: 'python3 rnnlm.py'
