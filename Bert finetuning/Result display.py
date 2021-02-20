import os
import torch
from sklearn.metrics import f1_score


def get_result(CV_type, sample_type):
    y_pred = []
    y_true = []
    x=0
    for i in range(10):
        load_path = CV_type + os.path.sep + sample_type + os.path.sep + 'Dataset_KFold_' + str(
            i) + os.path.sep + 'evaluate' \
                    + os.path.sep + 'evaluate.pth'
        state_dict = torch.load(load_path)
        x += f1_score(state_dict['y_true'], state_dict['y_pred'], average = 'weighted')

    print('F1 - score : ' + sample_type + ' ' +str(x/10))


get_result('Temp_Data_Files', 'OverSampled')
get_result('Temp_Data_Files_OS_NS_US_summary', 'OverSampled')