import os

MODEL_NAME = "ProteinBERT"
ADDED_TOKENS_PER_SEQ = 2
SEQ_CUTOFF = 40
MODEL_PATH = './dev/models'
DATA_PATH = './dev/data(test)'
ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<OTHER>', '<START>', '<END>', '<PAD>']

NEG_TST_PATH = os.path.join(DATA_PATH, 'test_sets', "negative")
POS_TST_PATH = os.path.join(DATA_PATH, 'test_sets', "positive")
COV_POS_TST_PATH = os.path.join(DATA_PATH, 'test_sets', "coords")
PREDS_PATH = "./dev/results"
SEP = "\t"

MARKER_SCALE_ALL = 4
MARKER_SCALE_POS_ONLY = 40

CUTOFF_VALUE = 0.5

TST_SETS_FILEPATHS = [
    os.path.join(POS_TST_PATH, "csga2203_nr40.fa"),
    os.path.join(POS_TST_PATH, "csga2203ER4_nr40.fa")
]

COV_TST_SETS_FILEPATHS_PAIRS = [
    (os.path.join(POS_TST_PATH, "bass_domain", "bass_ntm_domain_test.fa"), os.path.join(COV_POS_TST_PATH, "bass_ntm_domain_test_coord.csv")),
    (os.path.join(POS_TST_PATH, "fass_domain", "fass_ntm_domain_test.fa"), os.path.join(COV_POS_TST_PATH, "fass_ntm_domain_test_coord.csv")),
    (os.path.join(POS_TST_PATH, "fass_domain", "fass_ntm_domain_test.fa"), os.path.join(COV_POS_TST_PATH, "fass_ntm_domain_test_coord_v2.csv"))
]


def makedir(dirpath: str) -> str:
    if len(os.path.split(dirpath)) > 1:
        directory = os.path.dirname(dirpath)
    else:
        directory = dirpath
    os.makedirs(directory, exist_ok=True)
    return dirpath
