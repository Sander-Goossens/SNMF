#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:38 2019

@author: mishugeb
"""
from src.models.S_NMF.SigProfilerExtractor.SigProfilerExtractor import sigpro as sig
def main():
    data = sig.importdata("text")
    N = 10
    out = "C:/Users/sande/PycharmProjects/MEP/data/processed/bootstrapped_sameSplit/N_{}".format(N)
    data = out + '/X1_train.text'
    label = out + '/Y1_train.text' #one-hot encoded
    sig.sigProfilerExtractor("text", "example_output", data, label, minimum_signatures=5, maximum_signatures=6, nmf_replicates=1)

if __name__ == '__main__':
    main()
