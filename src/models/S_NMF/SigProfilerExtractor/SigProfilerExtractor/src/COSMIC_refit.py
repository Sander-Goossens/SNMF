
import numpy as np
import pandas as pd
from src.models.S_NMF.SigProfilerExtractor.SigProfilerExtractor import plotActivity as plot_ac


input_path = "C:/Users/sande/PycharmProjects/MEP/src/models/S_NMF/SigProfilerExtractor/SigProfilerExtractor/src/CV/final_test_new/K7_c1_p0_reps10_f_all/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Activities/SBS96_De-Novo_Activities.txt"
output_path = "C:/Users/sande/PycharmProjects/MEP/src/models/S_NMF/SigProfilerExtractor/SigProfilerExtractor/src/CV/final_test_new/K7_c1_p0_reps10_f_all/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Activities/"

# originalProcessAvg=pd.DataFrame(processAvg, index=index)
# originalProcessAvg.columns = listOfSignatures

# sub.signature_decomposition()
# final_signatures = sub.signature_decomposition(processAvg, m, layer_directory2, genome_build=genome_build, cosmic_version=cosmic_version, add_penalty=0.05, remove_penalty=0.01, mutation_context=mutation_context, make_decomposition_plots=make_decomposition_plots, originalProcessAvg=originalProcessAvg)

if signatures.shape[0] == 96:
    sigDatabase = pd.read_csv(paths + "/data/Reference_Signatures/" + genome_build + "/COSMIC_v" + str(
        cosmic_version) + "_SBS_" + genome_build + ".txt", sep="\t", index_col=0)

    signames = sigDatabase.columns
check_rule_penalty=1.0

_, exposures, L2dist, similarity, kldiv, correlation, cosine_similarity_with_four_signatures = ss.add_remove_signatures(
    sigDatabase,
    signatures[:, i],
    metric="l2",
    solver="nnls",
    background_sigs=[0, 4],
    permanent_sigs=[0, 4],
    candidate_sigs="all",
    allsigids=signames,
    add_penalty=add_penalty,
    remove_penalty=remove_penalty,
    check_rule_negatives=check_rule_negatives,
    checkrule_penalty=check_rule_penalty,
    directory=directory + "/Solution_Stats/Cosmic_" + mutation_context + "_Decomposition_Log.txt",
    connected_sigs=connected_sigs,
    verbose=False)