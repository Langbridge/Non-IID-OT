import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, ACSEmployment
from lib.evals import eval_report

import sys
sys.path.insert(0, "/Users/al4518/Desktop/PhD/FPD-OT/FPD-OT/COT")
from repair import StoppingRepair

def post_2018_relp_mapping(df: pd.DataFrame):
    relp_mapping = {
        20: 0,
        21: 1, # distinction between same-sex and opposite-sex partnerships
        22: 13, # distinction between same-sex and opposite-sex partnerships
        23: 1,
        24: 13,
        25: 2,
        26: 3,
        27: 4,
        28: 5,
        29: 6,
        30: 7,
        31: 8,
        32: 9,
        33: 10,
        34: 12, # roomer or boarder category disappeared
        35: 14,
        36: 15,
        37: 16,
        38: 17
    }

    assert 'RELSHIPP' in df.columns, "Must only be used for post-2018 data with modified RELSHIPP column."
    df['RELP'] = df['RELSHIPP'].map(relp_mapping)

    return df

def load_ACS_income(year: int, state: str, encode=True):
    # load data
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    data = data_source.get_data(states=[state], download=True)

    # relshipp mapping
    if year > 2018:
        data = post_2018_relp_mapping(data)
        
    features, labels, _ = ACSIncome.df_to_pandas(data)

    if encode:
        # U,S encodings
        features['RAC1P'] = (features['RAC1P'] == 1.0).astype(int) # white = 1.0
        features['SEX'] = (features['SEX'] == 1.0).astype(int) # male = 1.0

    return features, labels

def preprocess_ACS_data(data: pd.DataFrame, u='SCHL', s='RAC1P', u_thresh=1.0, s_thresh=1.0):
    data['u'] = (data[u] >= u_thresh).astype(int)
    data['s'] = (data[s] >= s_thresh).astype(int)

    data = data.drop(columns=[u, s])

    return data

def repair_year_state(stopping_data: pd.DataFrame, other_data: pd.DataFrame):
    pass
#     assert ['u', 's'] in stopping_data.columns, "Data must contain columns 'u' and 's' to be repaired."

#     for f in stopping_data.feat
#     repair_operation = StoppingRepair(stopping_data)
#     repaired_stopping_data = repair_operation.repair(data)

#     report = eval_report(data, repaired_data)

#     return repaired_data, report

