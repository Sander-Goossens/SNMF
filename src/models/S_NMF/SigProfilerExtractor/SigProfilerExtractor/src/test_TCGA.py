#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:45:38 2019

@author: mishugeb
"""
from src.models.S_NMF.SigProfilerExtractor.SigProfilerExtractor import sigpro as sig
import numpy as np
def main():
    # data = sig.importdata("text")

    min_sig = 5
    max_sig = 5
    reps = 10
    lr = 5e-3

    l_c = [0.0, 0.1]

    l_p = [0.0001]

    folds = ['_all']

    cancers =  ['BRCA', 'COAD', 'UCEC','PRAD','STAD', 'OV']
    # cancers =  ['BRCA', 'OV']
    # cancers =  ['COAD', 'UCEC','STAD']

    # cosmic = 'S3'

    for cancer in cancers:
        for fold in folds:
            test_path = "C:/Users/sande/PycharmProjects/MEP/data/processed/Volkova/filtered"
            # # test_path = "C:/Users/sande/PycharmProjects/MEP/data/processed/Volkova/filtered_mean"
            # test_data = test_path + '/X_test{}.text'.format(fold)
            test_label = test_path + '/Y_test{}.text'.format(fold)  # one-hot encoded


            test_path = 'C:/Users/sande/PycharmProjects/MEP_data/data/TCGA/profiles/norm_SBS/'
            # test_path = "C:/Users/sande/PycharmProjects/MEP/data/processed/Volkova/filtered_mean"
            test_data = test_path + '{}.text'.format(cancer)
            # test_label = test_path + '/Y_test{}.text'.format(fold)  # one-hot encoded

            seed_path = "CV/Seeds_2.txt"

            for k in range(min_sig,max_sig+1):
                # Acc_train ; F1_train ; Rec_train ; Lce ; Ltot ; Epochs ; Stability_avg ; Stability_min ; Acc_refit ; F1_refit ; Lrec_refit ; Lce_refit ; Ltot_refit ; Epoch_refit ; Acc_test ; F1_test ; Rec_test
                results = np.zeros((len(l_c), len(l_p), 31))

                for c_idx, lambda_c in enumerate(l_c):
                    c_idx = c_idx
                    for p_idx, lambda_p in enumerate(l_p):
                        # model_path = 'CV/final_test_3/SNMF_K5_c{}_p0_reps10_f_all_COSMIC_{}'.format(c_idx, cosmic)
                        # model_path = 'CV/final_test_3/SNMF_K5_c{}_p0_reps10_f_all_COSMIC_{}'.format(c_idx)
                        model_path = 'CV/final_test_3/SNMF_K5_c{}_p0_reps10_f_all'.format(c_idx)

                        # output_path = 'CV/final_test_volkova_mean/SNMF_K5_c1/'
                        # output_path = 'TCGA_test/{}/SNMF_K5_c{}_COSMIC_{}/'.format(cancer, c_idx, cosmic)
                        # output_path = 'TCGA_test/{}/SNMF_K5_c{}_COSMIC_{}/'.format(cancer, c_idx)
                        output_path = 'TCGA_test/{}/SNMF_K5_c{}/'.format(cancer, c_idx)



                        # output_path = 'CV/final/K{}_c{}_p{}_reps{}_f{}'.format(k, c_idx, p_idx, reps, fold)


                        # TRAINING
                        # results[c_idx, p_idx, :25] = sig.sigProfilerExtractor("text", output_path, train_data, train_label, minimum_signatures=k, maximum_signatures=k, seeds = seed_path, nmf_replicates=reps, lambda_c= lambda_c, lr= lr, lambda_p= lambda_p, make_decomposition_plots=False)

                    # TEST / VALIDATION
                        # UNFILTERED
                        # results[c_idx, p_idx, 25:28]= sig.test_sigProfilerExtractor("text", output_path, test_data, test_label, minimum_signatures=k, maximum_signatures=k, nmf_replicates=reps, lambda_c= float(lambda_c), lr= float(lr), lambda_p= float(lambda_p), filter = False, model_path=model_path, cancer=cancer, cosmic=cosmic)
                        results[c_idx, p_idx, 25:28]= sig.test_sigProfilerExtractor("text", output_path, test_data, test_label, minimum_signatures=k, maximum_signatures=k, nmf_replicates=reps, lambda_c= float(lambda_c), lr= float(lr), lambda_p= float(lambda_p), filter = False, model_path=model_path, cancer=cancer)


                        # FILTERED
                        print(results[c_idx, p_idx, 6])
                        if results[c_idx, p_idx, 6] == 0:
                            results[c_idx, p_idx, 28:] = results[c_idx, p_idx, 25:28]
                        else:
                            print('ERROR: filter')
                            results[c_idx, p_idx, 28:]= sig.test_sigProfilerExtractor("text", output_path, test_data, test_label, minimum_signatures=k, maximum_signatures=k, nmf_replicates=reps, lambda_c= float(lambda_c), lr= float(lr), lambda_p= float(lambda_p), filter = True)

                        # f = 'SNMF_final_test_.txt'
                        f = 'TCGA_test/{}/SNMF_k{}_f{}.txt'.format(cancer, k, fold)


                        with open(f, 'a') as outfile:
                            outfile.write('\n # Lc = {} \t L2 = {} \n'.format(l_c[c_idx], l_p[p_idx]))
                            np.savetxt(outfile, results[c_idx, p_idx,:], fmt='%-8.5f', newline= ' ')
                            outfile.close()


if __name__ == '__main__':
    main()
