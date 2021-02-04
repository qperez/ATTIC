### IMPORT Libraries ###
import matplotlib.pyplot as plt
import pandas as pd
import torch
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
### END IMPORT Libraries ###

# TODO: device étant une constante pour la clareté du message preférer les une font MAJ
# TODO: DEVICE = ...
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

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

train, valid, test = TabularDataset.splits(path='', train='train.csv', validation='valid.csv',
                                           test='test.csv', format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=10, device=device, train=True)
valid_iter = BucketIterator(valid, batch_size=10, device=device, train=True)
test_iter = Iterator(test, batch_size=10, device=device, train=False, shuffle=False, sort=False)


# TODO: même si la classe est toute petite il serait préférable de la placer dans une classe python à part
# TODO: cette classe
class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

### BEGIN Functions ####
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


def load_metrics(load_path):
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# Training Function
def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=5,
          eval_every=len(train_iter) // 2,
          file_path='',
          best_valid_loss=float("Inf")):

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs)
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
                        # TODO: peut etre factorisé en :
                        # TODO: summarydescription.type(torch.LongTensor).to(device)
                        # TODO: cela vous évite la réaffection
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
                    save_checkpoint('model' + '/' + 'model.pth', model, best_valid_loss)
                    save_metrics('metrics' + '/' + 'metrics.pth', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics('metrics' + '/' + 'metrics.pth', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

# Evaluation Function

def evaluate(model, test_loader):
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

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['NBug', 'Bug'])
    ax.yaxis.set_ticklabels(['NBug', 'Bug'])
### END Functions ###

# TODO : évitez de mixer des bouts de main au milieu de 2 fonctions
# TODO : j'ai volontairement replacé la fonction evalutae() plus haut
model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)


train_loss_list, valid_loss_list, global_steps_list = load_metrics('metrics'+'/'+'/metrics.pth')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()


# TODO: pour avoir quelque chose de vraiment jolie n'hésitez pas à utiliser le main pythonien
#if __name__ == '__main__':
#    mon code du programme principal.....

best_model = BERT().to(device)

load_checkpoint('model' + '/' + 'model.pth', best_model)

evaluate(best_model, test_iter)