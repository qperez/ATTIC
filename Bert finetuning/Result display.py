import os
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def get_result(CV_type, sample_type, show_trainning_graph=False,show_confusion_matrix=False):
    y_pred = []
    y_true = []
    f1 = 0
    acc = 0
    for i in range(10):
        load_path = CV_type + os.path.sep + sample_type + os.path.sep + 'Dataset_KFold_' + str(
            i) + os.path.sep + 'evaluate' \
                    + os.path.sep + 'evaluate.pth'
        load_path_alt = CV_type + os.path.sep + sample_type + os.path.sep + 'Dataset_KFold_' + str(
            i) + os.path.sep + 'metrics' \
                        + os.path.sep + 'metrics.pth'
        state_dict = torch.load(load_path)
        state_dict_alt = torch.load(load_path_alt)
        f1 += f1_score(state_dict['y_true'], state_dict['y_pred'], average='weighted')
        acc += accuracy_score(state_dict['y_true'], state_dict['y_pred'], normalize=True)

        cm = confusion_matrix(state_dict['y_true'], state_dict['y_pred'], labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])


        train_loss_list, valid_loss_list, global_steps_list = state_dict_alt['train_loss_list'], \
                                                              state_dict_alt['valid_loss_list'], \
                                                              state_dict_alt['global_steps_list']

        if show_confusion_matrix:
            ax = plt.subplot()
            sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt="d")
            plt.show()
            ax.set_title('Confusion Matrix')

            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')

            ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
            ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
            plt.show()
        if show_trainning_graph:
            plt.plot(global_steps_list, train_loss_list, label='Train')
            plt.plot(global_steps_list, valid_loss_list, label='Valid')
            plt.xlabel('Global Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
    print('F1 - score : ' + CV_type + sample_type + ' ' + str(f1 / 10))
    print('Accuracy : ' + CV_type + sample_type + ' ' + str(acc / 10))


#get_result('Temp_Data_Files', 'OverSampled', show_trainning_graph=True)
print("----------------")
#get_result('Temp_Data_Files_OS_NS_US_summary', 'OverSampled', show_trainning_graph=True)
#get_result('Temp_Data_Files_OS_NS_US_summary', 'UnderSampled', show_trainning_graph=True)
#get_result('Temp_Data_Files_OS_NS_US_summary', 'NotSampled', show_trainning_graph=True)
print("----------------")
get_result('Temp_Data_Files_Multiclass', 'OverSampled',show_confusion_matrix=True)
# get_result('Temp_Data_Files_Multiclass', 'UnderSampled')
# get_result('Temp_Data_Files_Multiclass', 'NotSampled')
