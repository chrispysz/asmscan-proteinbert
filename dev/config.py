import os

MODEL_NAME = "ProteinBERT"
ADDED_TOKENS_PER_SEQ = 2
SEQ_CUTOFF = 40
MODEL_PATH = 'models_no_freeze'
DATA_PATH = 'data(test)'
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

NEG_TST_PATH = os.path.join(DATA_PATH, 'test_sets', "negative")
POS_TST_PATH = os.path.join(DATA_PATH, 'test_sets', "positive")
PREDS_PATH = "results_no_freeze"
SEP = "\t"

MARKER_SCALE = 2

TST_SETS_FILEPATHS = [
    os.path.join(NEG_TST_PATH, "DisProt", "DisProt_test.fa"),
    os.path.join(NEG_TST_PATH, "NLReff", "NLReff_test.fa"),
    os.path.join(NEG_TST_PATH, "PB40", "PB40_1z20_clu50_test.fa"),
    os.path.join(NEG_TST_PATH, "PB40", "PB40_1z20_clu50_test_sampled10000.fa"),
    os.path.join(POS_TST_PATH, "bass_domain", "bass_ntm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "bass_domain", "bass_other_ctm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "bass_domain", "bass_other_ntm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "bass_motif", "bass_ntm_motif_test.fa"),
    os.path.join(POS_TST_PATH, "bass_motif", "bass_ntm_motif_env5_test.fa"),
    os.path.join(POS_TST_PATH, "bass_motif", "bass_ntm_motif_env10_test.fa"),
    os.path.join(POS_TST_PATH, "fass_domain", "fass_ctm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "fass_domain", "fass_ntm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "fass_domain", "het-s_ntm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "fass_domain", "pp_ntm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "fass_domain", "sigma_ntm_domain_test.fa"),
    os.path.join(POS_TST_PATH, "fass_motif", "fass_ctm_motif_test.fa"),
    os.path.join(POS_TST_PATH, "fass_motif", "fass_ctm_motif_env5_test.fa"),
    os.path.join(POS_TST_PATH, "fass_motif", "fass_ctm_motif_env10_test.fa"),
    os.path.join(POS_TST_PATH, "fass_motif", "fass_ntm_motif_test.fa"),
    os.path.join(POS_TST_PATH, "fass_motif", "fass_ntm_motif_env5_test.fa"),
    os.path.join(POS_TST_PATH, "fass_motif", "fass_ntm_motif_env10_test.fa")
]


def makedir(dirpath: str) -> str:
    if len(os.path.split(dirpath)) > 1:
        directory = os.path.dirname(dirpath)
    else:
        directory = dirpath
    os.makedirs(directory, exist_ok=True)
    return dirpath
