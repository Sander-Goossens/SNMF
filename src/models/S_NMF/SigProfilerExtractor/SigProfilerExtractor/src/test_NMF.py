#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:38 2019

@author: mishugeb
"""
from src.models.S_NMF.SigProfilerExtractor.SigProfilerExtractor import sigpro as sig
import numpy as np
import random

random.seed(10)

def main():
    # data = sig.importdata("text")

    min_sig = 5
    max_sig = 5
    reps = 10
    lr = 5e-3

    # l_c = [0., 0.01, 0.1, 0.5]
    # l_p = [0., 0.00001, 0.001, 0.05]
    # l_c = [0., 1e-3]
    # l_p = [0., 1e-3]

    # l_c = [0., 0.001, 0.01, 0.05,  0.1, 0.25, 0.5]
    # l_p = [0., 0.00001, 0.0001, 0.001, 0.01]

    # l_c = [1.0]
    # l_p = [0., 0.00001, 0.0001, 0.001, 0.01]

    # l_c = [0.1]
    # l_p =[0.001]

    # folds = [2]
    folds = ['_all']

    output_path = "CV/final/K5_c1_p001"
    seed_path = "CV/Seeds.txt"

    for fold in folds:
        N_train = 100
        train_path = "C:/Users/sande/PycharmProjects/MEP/data/processed/bootstrapped_sameSplit/N_{}".format(N_train)
        train_data = train_path + '/X_train{}.text'.format(fold)
        train_label = train_path + '/Y_train{}.text'.format(fold)  # one-hot encoded

        N_test = 1000
        test_path = "C:/Users/sande/PycharmProjects/MEP/data/processed/bootstrapped_sameSplit/N_{}".format(N_test)
        test_data = test_path + '/X_test{}.text'.format(fold)
        test_label = test_path + '/Y_test{}.text'.format(fold)  # one-hot encoded


        # Acc_train ; F1_train ; Rec_train ; Lce ; Ltot ; Epochs ; Stability_avg ; Stability_min ; Acc_refit ; F1_refit ; Lrec_refit ; Lce_refit ; Ltot_refit ; Epoch_refit ; Acc_test ; F1_test ; Rec_test
        results = np.zeros((len(l_c), len(l_p), 18))

        for c_idx, lambda_c in enumerate(l_c):
            for p_idx, lambda_p in enumerate(l_p):

                # output_path = "training_Lc{}_Lp{}_lr{}_rep{}_min{}_max{}".format(lambda_p, lambda_p, lr, reps, min_sig, max_sig)
                # output_path = "train_K{}_c{}_p{}_reps{}_f{}".format(k, c_idx, p_idx, reps, fold)

                # output_path = "test_K{}_c{}_p{}_reps{}_f{}".format(k, c_idx, p_idx, reps, fold)
                # output_path = "Real_only"

                # output_path = 'K{}_c{}_p{}_reps{}_f{}'.format(k, c_idx, p_idx, reps, fold)
                #TRAINING
                results[c_idx, p_idx, :15] = sig.sigProfilerExtractor("text", output_path, train_data, train_label,seeds = seed_path,  minimum_signatures=min_sig, maximum_signatures=max_sig, nmf_replicates=reps, lambda_c= lambda_c, lr= lr, lambda_p= lambda_p, make_decomposition_plots=False)


                #Test
                results[c_idx, p_idx, 15:]= sig.test_sigProfilerExtractor("text", output_path, test_data, test_label, minimum_signatures=5, maximum_signatures=5, nmf_replicates=reps, lambda_c= float(lambda_c), lr= float(lr), lambda_p= float(lambda_p))

        f = output_path+'.txt'
        # f = 'test_all_01.txt'

        with open(f, 'w') as outfile:
            outfile.write('# Array shape: {0}\n'.format(results.shape))

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            i = 0
            for data_slice in results:
                outfile.write('# Lc = {} ; Ntrain = {}\n'.format(l_c[i], N_train))


                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 8 characters in width
                # with 5 decimal places.
                np.savetxt(outfile, data_slice, fmt='%-8.5f')

                # Writing out a break to indicate different slices...
                # outfile.write('# New slice\n')
                i += 1


            # new_data = np.loadtxt(f)

            # new_data = new_data.reshape(results.shape)
            # assert np.all(new_data == data)

if __name__ == '__main__':
    main()
