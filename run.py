import sys
import pandas as pd
import torch
import numpy as np
import sklearn
import nltk
from nltk import word_tokenize
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader as api

torch.manual_seed(57)

hw1_train = sys.argv[1]
hw1_test = sys.argv[2]
submission = sys.argv[3]

df_train = pd.read_csv(hw1_train)
df_test = pd.read_csv(hw1_test)

x = df_train['UTTERANCES'].values
y = df_train['CORE RELATIONS'].values
x_testing = df_test['UTTERANCES'].values

# creating a list of all possible y values
values = []
for item in y:
    if isinstance(item, str):
        items = item.split(" ")
        for new_item in items:
            if new_item not in values:
                values.append(new_item)

values.sort()

# creating an array of all labels where each row is a list of zeros and ones corresponding to the true values
y_labels = []
for item in y:
    if isinstance(item, str):
        items = item.split(" ")
        zero_list = [0.0] * 19
        for new_item in items:
            zero_list[values.index(new_item)] = 1.0
        y_labels.append(zero_list)
    else:
        new_item = "none"
        zero_list = [0.0] * 19
        zero_list[values.index(new_item)] = 1.0
        y_labels.append(zero_list)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(3, 3), analyzer="char", max_features=3000)
#vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english', max_features=3000)
vectorizer.fit(x)
x_vec = torch.FloatTensor(vectorizer.transform(x).toarray())
y_vec = torch.FloatTensor(y_labels)
x_testing_vec = torch.FloatTensor(vectorizer.transform(x_testing).toarray())

# add glove embeddings
nltk.download('punkt_tab')
wv = api.load('glove-twitter-200')

text_embeddings = []
for text in df_train['UTTERANCES']:
    token_embeddings = []
    for token in word_tokenize(text):
        if token in wv:
            token_embeddings.append(wv[token])
    text_embedding = np.array(token_embeddings).mean(axis=0)
    text_embeddings.append(text_embedding)
x_embed = np.array(text_embeddings) 

text_embeddings = []
for text in df_test['UTTERANCES']:
    token_embeddings = []
    for token in word_tokenize(text):
        if token in wv:
            token_embeddings.append(wv[token])
    if token_embeddings == []:
        text_embedding = np.array([0]*200)
    else:
        text_embedding = np.array(token_embeddings).mean(axis=0)
    text_embeddings.append(text_embedding)
x_testing_embed = np.array(text_embeddings)

# data normalization
mean, std = x_vec.mean(), x_vec.std()
x_vec = (x_vec - mean) / std
x_testing_vec = (x_testing_vec - mean) / std

x_embed = torch.tensor(x_embed)
x_testing_embed = torch.tensor(x_testing_embed)

x_vec = torch.cat((x_vec, x_embed), dim=1)
x_testing_vec = torch.cat((x_testing_vec, x_testing_embed), dim=1)

# train test split
x_train_vec, x_test_vec, y_train_vec, y_test_vec = sklearn.model_selection.train_test_split(x_vec, y_vec, test_size=0.15, random_state=64)

#model definition
class RelationExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 19)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x): # x is (batch_size, input_dim)
        x = self.relu(self.layer1(x)) # (batch_size, 256)
        x = self.dropout(x)
        x = self.relu(self.layer2(x)) # (batch_size, 256)
        x = self.dropout(x)
        x = self.relu(self.layer3(x)) # (batch_size, 256)
        x = self.dropout(x)
        x = self.layer4(x) # (batch_size, 19)
        return x # (batch_size,)

# create batches for training and test
train_dataset = TensorDataset(x_train_vec, y_train_vec)
test_dataset = TensorDataset(x_test_vec, y_test_vec)
train_loader = DataLoader(train_dataset, batch_size=32, pin_memory=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True, shuffle=False)

model = RelationExtractor(x_train_vec.size(1))

# define loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.00003)

# training 
def training(path):
    epochs = 150
    best_val_loss = torch.tensor(1.)
    for epoch in range(epochs):
        
        running_loss = torch.tensor(0.)
        for x_batch, y_batch in train_loader:

            # forward pass
            model.train()
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)

            running_loss += loss

            # backward pass
            loss.backward()
            optimizer.step()

        # evaluate on test set
        model.eval()
        val_running_loss = torch.tensor(0.)
        with torch.no_grad():

            correct_predictions = 0
            for x_batch, y_batch in test_loader:
                output = model(x_batch)
                val_loss = loss_fn(output, y_batch)
                val_running_loss += val_loss
                index_row = 0
                index_column = 0
                for row in output:
                    row_accuracy = []
                    index_column = 0
                    for i in row:
                        if i > 0.0:
                            if y_batch[index_row, index_column] == 1.0:
                                row_accuracy.append(True) # correct class prediction
                            else:
                                row_accuracy.append(False) # incorrect prediction that an item is not part of a class

                        else:
                            if y_batch[index_row, index_column] == 0.0:
                                row_accuracy.append(True) # correct prediction that an item is not part of a class
                            else:
                                row_accuracy.append(False) # incorrect class prediction

                        index_column += 1
                    if False not in row_accuracy:
                        correct_predictions += 1
                    index_row += 1


            acc = (correct_predictions / len(test_dataset))
            if (val_running_loss / len(test_loader)) < best_val_loss :
                torch.save(model.state_dict(), path)
                best_val_loss = (val_running_loss / len(test_loader))
            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_running_loss / len(test_loader)}, Acc: {acc}')

# Inputting testing data to create predictions
def Use_model(model, x):
    predictions = []
    model.eval()
    with torch.no_grad():
        output = model(x)
        index_row = 0
        index_column = 0
        for row in output:
            row_accuracy = []
            index_column = 0
            most_likely = -999.9
            most_likely_index = 0
            for i in row:
                if i > 0.0:
                    if output[index_row, index_column] > 0.0:
                        row_accuracy.append(values[index_column])
                if i > most_likely:
                    most_likely = i
                    most_likely_index = index_column
                index_column += 1
            if row_accuracy == []: # if no class is predicted (which is different from the none class), predict the class with the largest value in the tensor
                predictions.append([x_testing[index_row], [values[most_likely_index]]])
            else:
                predictions.append([x_testing[index_row], row_accuracy])
            index_row += 1
        return predictions

training("best_model.pt")
best_model = RelationExtractor(x_train_vec.size(1))
best_model.load_state_dict(torch.load("best_model.pt", weights_only=True))

predictions = Use_model(best_model, x_testing_vec.float())

# create csv of outputs from the model on the hw1_test.csv dataset
csv_list = []
label = 0
for prediction in predictions:
    if prediction[1] == [] or prediction[1][0] == "none": # if there are no classes predicted, predict none
        row = [str(label), "none"]
        csv_list.append(row)
    elif len(prediction[1]) == 1: # if there is one class prediction, predict it
        row = [str(label), str(prediction[1][0])]
        csv_list.append(row)
    else: # predict multiple classes
        multi_pred = ""
        for pred in prediction[1]:
            if str(pred) != "none":
                multi_pred = multi_pred + str(pred) + " "
        multi_pred = multi_pred[:-1]
        row = [str(label), multi_pred]
        csv_list.append(row)
    label += 1

for_csv = pd.DataFrame(csv_list, columns = ["ID", "Core Relations"])
for_csv.to_csv(submission, index = False)

