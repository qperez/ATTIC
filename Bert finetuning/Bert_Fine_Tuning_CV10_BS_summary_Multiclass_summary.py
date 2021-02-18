0  # Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from os import path
import codecs
import json
import os
import numpy
import shutil
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

#######################################################################################################################
#######################################################################################################################

######################################## DATA-SET CREATION ############################################################


boost_summary = 3
project_keys = ["HTTPCLIENT", "LUCENE", "JCR"]
random_seed = 42

data_directory = "Temp_Data_Files_Multiclass"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)


def load_data():
    raw_data = []
    data_directory = ".." + os.path.sep + "data"
    for filename in os.listdir(data_directory):
        with codecs.open(data_directory + os.path.sep + filename, "r", "utf-8") as fin:
            raw_data += json.load(fin)
    return raw_data


def get_corpus_labels(raw_data):
    # Corpus building.
    corpus = []
    labels = []
    n_bug = 0
    for n_file in raw_data:
        # txt = ""
        # for i in range(boost_summary):
        # txt += n_file["summary"] + " "

        # corpus.append(txt + " " + n_file["description"])
        corpus.append(n_file["summary"])
        labels.append(n_file["classified"])
    return corpus, labels


def Undersampling(dataFrame):
    X = np.array(dataFrame[dataFrame.columns[1]])
    y = np.array(dataFrame["label"])
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    df = pd.DataFrame({'label': np.array(y_res.flatten()), 'summarydescription': np.array(X_res.flatten())})
    return df


def Oversampling(dataFrame):
    X = np.array(dataFrame[dataFrame.columns[1]])
    y = np.array(dataFrame["label"])
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    df = pd.DataFrame({'label': np.array(y_res.flatten()), 'summarydescription': np.array(X_res.flatten())})
    return df


def create_files_for_Kfold(nb_csv, df_test, SamplingMode):
    if SamplingMode == 0:
        data_directory = "Temp_Data_Files_Multiclass" + os.path.sep + "NotSampled"
    if SamplingMode == 1:
        data_directory = "Temp_Data_Files_Multiclass" + os.path.sep + "UnderSampled"
    if SamplingMode == 2:
        data_directory = "Temp_Data_Files_Multiclass" + os.path.sep + "OverSampled"

    for i in range(nb_csv):
        if not os.path.exists(data_directory + os.path.sep + "Dataset_KFold_" + str(i)):
            os.makedirs(data_directory + os.path.sep + "Dataset_KFold_" + str(i))
            os.makedirs(data_directory + os.path.sep + "Dataset_KFold_" + str(i) + os.path.sep + "model")
            os.makedirs(data_directory + os.path.sep + "Dataset_KFold_" + str(i) + os.path.sep + "metrics")
            os.makedirs(data_directory + os.path.sep + "Dataset_KFold_" + str(i) + os.path.sep + "evaluate")
            df_test.to_csv(data_directory + os.path.sep + "Dataset_KFold_" + str(i) + os.path.sep + "test.csv",
                           index=False)

    csv_file_list = []
    for i in range(nb_csv):
        csv_file_list.append(
            data_directory + os.path.sep + "Kfold_csv_files" + os.path.sep + "data_fold_" + str(i) + ".csv")

    for i in range(nb_csv):

        list_of_dataframes = []

        for filename in csv_file_list:

            if filename[-5] != str(i):
                list_of_dataframes.append(pd.read_csv(filename))

            else:
                valid_df = pd.read_csv(filename)
                valid_df.to_csv(data_directory + os.path.sep + "Dataset_KFold_" + str(i) + os.path.sep + "valid.csv",
                                index=False)

        train_df = pd.concat(list_of_dataframes)
        if SamplingMode == 1:
            train_df = Undersampling(train_df)
        if SamplingMode == 2:
            train_df = Oversampling(train_df)
        train_df.to_csv(data_directory + os.path.sep + "Dataset_KFold_" + str(i) + os.path.sep + "train.csv",
                        index=False)

        del train_df


def create_csv_for_Kfold(corpus, labels, nb_csv=10, relabel=True, train_set_size=0.85, SamplingMode=0):
    print("starting creation of datasets for cross validation")
    data = {'label': labels, 'summmarydescription': corpus}
    df = pd.DataFrame(data)
    possible_labels = df.label.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['label'] = df.label.replace(label_dict)

    if SamplingMode == 0:
        print('creating dataset of : ' + str(df.shape))
        data_directory = "Temp_Data_Files_Multiclass" + os.path.sep + "NotSampled"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        data_directory = data_directory + os.path.sep + "Kfold_csv_files"

    if SamplingMode == 1:
        print('creating undersampled dataset of : ' + str(df.shape))
        data_directory = "Temp_Data_Files_Multiclass" + os.path.sep + "UnderSampled"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        data_directory = data_directory + os.path.sep + "Kfold_csv_files"

    if SamplingMode == 2:
        print('creating oversampled dataset of : ' + str(df.shape))
        data_directory = "Temp_Data_Files_Multiclass" + os.path.sep + "OverSampled"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        data_directory = data_directory + os.path.sep + "Kfold_csv_files"

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    df_train = df.sample(frac=train_set_size, random_state=random_seed)
    df_test = df.drop(df_train.index)

    file_name = data_directory + os.path.sep + "data_train.csv"
    df_train.to_csv(file_name, index=False)
    file_name = data_directory + os.path.sep + "data_test.csv"
    df_test.to_csv(file_name, index=False)

    sizes = []
    size = len(df_train.index) // nb_csv
    rest = len(df_train.index) % nb_csv
    for i in range(nb_csv):
        sizes.append(size)

    for i in range(rest):
        sizes[i] += 1

    for i in range(nb_csv):
        df_i = df_train.sample(n=sizes[i], random_state=random_seed)
        df_train = df_train.drop(df_i.index)

        file_name = data_directory + os.path.sep + "data_fold_" + str(i) + ".csv"
        df_i.to_csv(file_name, index=False)

    create_files_for_Kfold(nb_csv, df_test, SamplingMode)


corpus, labels = get_corpus_labels(load_data())
create_csv_for_Kfold(corpus, labels, SamplingMode=2)
create_csv_for_Kfold(corpus, labels, SamplingMode=0)
create_csv_for_Kfold(corpus, labels, SamplingMode=1)

#######################################################################################################################
#######################################################################################################################

############################################### BERT FINE-TUNING ######################################################
print("starting fine tuning")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def init(TRAIN_FILE, VALID_FILE, TEST_FILE):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model parameter
    MAX_SEQ_LEN = 128
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                       fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('label', label_field), ('summarydescription', text_field)]

    # TabularDataset

    train, valid, test = TabularDataset.splits(path='', train=TRAIN_FILE, validation=VALID_FILE,
                                               test=TEST_FILE, format='CSV', fields=fields, skip_header=True)

    # Iterators

    train_iter = BucketIterator(train, batch_size=10, device=device, train=True)
    valid_iter = BucketIterator(valid, batch_size=10, device=device, train=True)
    test_iter = Iterator(test, batch_size=10, device=device, train=False, shuffle=False, sort=False)

    return train_iter, valid_iter, test_iter


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=14)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea


# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def save_pred(save_path, y_pred, y_true):
    if save_path is None:
        return

    state_dict = {'y_pred': y_pred,
                  'y_true': y_true}

    torch.save(state_dict, save_path)
    print(f'Pred saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function

def train(model,
          optimizer,
          train_loader,
          valid_loader,
          num_epochs,
          file_path_model,
          file_path_metrics,
          criterion=nn.BCELoss(),
          best_valid_loss=float("Inf")):
    # initialize running values
    eval_every = len(train_loader) // 2
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (label, summarydescription), _ in train_loader:
            label = label.type(torch.LongTensor)
            label = label.to(device)
            summarydescription = summarydescription.type(torch.LongTensor)
            summarydescription = summarydescription.to(device)
            output = model(summarydescription, label)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (label, summarydescription), _ in valid_loader:
                        label = label.type(torch.LongTensor)
                        label = label.to(device)
                        summarydescription = summarydescription.type(torch.LongTensor)
                        summarydescription = summarydescription.to(device)
                        output = model(summarydescription, label)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path_model, model, best_valid_loss)
                    save_metrics(file_path_metrics, train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path_metrics, train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


# Evaluation Function

def evaluate(model, test_loader, file_path):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (label, summarydescription), _ in test_loader:
            label = label.type(torch.LongTensor)
            label = label.to(device)
            summarydescription = summarydescription.type(torch.LongTensor)
            summarydescription = summarydescription.to(device)
            output = model(summarydescription, label)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(label.tolist())

    save_pred(save_path=file_path, y_pred=y_pred, y_true=y_true)
    print('Classification Report :')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))


training_path = "Temp_Data_Files_Multiclass" + os.path.sep


def cross_val(training_path, dataset_type):
    training_path += (dataset_type + os.path.sep + "Dataset_KFold_")
    print("STARTING CROSS VALIDATION FOR " + dataset_type + " DATASET" + '\n')
    for i in range(10):
        print("STARTING WITH FOLD NB " + str(i) + '\n')
        model = BERT().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5)

        train_iter, valid_iter, test_iter = init(
            TRAIN_FILE=training_path + str(i) + os.path.sep + "train.csv",
            VALID_FILE=training_path + str(i) + os.path.sep + "valid.csv",
            TEST_FILE=training_path + str(i) + os.path.sep + "test.csv")
        train(model=model, optimizer=optimizer, train_loader=train_iter, valid_loader=valid_iter, num_epochs=5,
              file_path_metrics=training_path + str(i) + os.path.sep + "metrics" + os.path.sep + "metrics.pth",
              file_path_model=training_path + str(i) + os.path.sep + "model" + os.path.sep + "model.pth")

        train_loss_list, valid_loss_list, global_steps_list = load_metrics(
            training_path + str(i) + os.path.sep + "metrics" + os.path.sep + "metrics.pth")

        plt.plot(global_steps_list, train_loss_list, label='Train')
        plt.plot(global_steps_list, valid_loss_list, label='Valid')
        plt.xlabel('Global Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        best_model = BERT().to(device)

        load_checkpoint(training_path + str(i) + os.path.sep + "model" + os.path.sep + "model.pth", best_model)

        evaluate(model=best_model, test_loader=test_iter,
                 file_path=training_path + str(i) + os.path.sep + "evaluate" + os.path.sep + "evaluate.pth")


cross_val(training_path, 'NotSampled')
cross_val(training_path, 'UnderSampled')
cross_val(training_path, 'OverSampled')
