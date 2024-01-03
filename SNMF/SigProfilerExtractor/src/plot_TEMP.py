
import numpy as np
import pandas as pd
from src.models.S_NMF.SigProfilerExtractor.SigProfilerExtractor import plotActivity as plot_ac


input_path = "C:/Users/sande/PycharmProjects/MEP/src/models/S_NMF/SigProfilerExtractor/SigProfilerExtractor/src/CV/final_test_new/K7_c1_p0_reps10_f_all/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Activities/SBS96_De-Novo_Activities.txt"
output_path = "C:/Users/sande/PycharmProjects/MEP/src/models/S_NMF/SigProfilerExtractor/SigProfilerExtractor/src/CV/final_test_new/K7_c1_p0_reps10_f_all/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Activities/"



plot_ac.plotActivity_real(input_path, output_file = output_path + "NMF_Activity_Plots_real_reordered.pdf", bin_size=50, log=False)

# # byte_plot = sp.run_PlotDecomposition(originalProcessAvg[denovo_cols], denovo_name, sigDatabases_DF[basis_cols],basis_names, weights, nonzero_exposures / 5000, directory, "test", mtype_par)
# input_sig
# output_sig
#
# plot.plotSBS(signature_subdirectory + "/" + mutation_type + "_S" + str(i) + "_Signatures" + ".txt",
#              signature_subdirectory + "/Signature_plot", "S" + str(i), m, True, custom_text_upper=stability_list,
#              custom_text_middle=total_mutation_list)
