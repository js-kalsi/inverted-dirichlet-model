import os

DATASET_PATH = os.getenv('DATASET_PATH') or './datasets/cm1.csv'
K = int(os.getenv('K') or 2)
CLUSTER_DROP_VAL = float(os.getenv('CLUSTER_DROP_VAL') or 0.001)
ALG_COVERAGE = float(os.getenv('ALG_COVERAGE') or 0.01)
THRESHOLD = int(os.getenv('THRESHOLD') or 100)
