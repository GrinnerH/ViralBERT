import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import transformers
import plotly.express as px
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from fast_ml.model_development import train_valid_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
def scores(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    return {
        "f1 score": f1_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "accuracy": balanced_accuracy_score(y_test, y_pred),
        "plot": disp.plot(),
    }

path = "."
reduced_path = os.path.join(path, "reducedDataset")
tweets_df = pd.read_csv(os.path.join(reduced_path, "all.csv"), header=0)
tweets_df.drop_duplicates(subset="text", inplace=True)
tweets_df["text_len"] = tweets_df["text"].apply(len)
tweets_df["verified"] = tweets_df["verified"].astype(bool)
tweets_df.info()

RT_THRESHOLDS = [0, 1, 20] # virality classes - change to use other thresholds
def vir_transform(rt, th=RT_THRESHOLDS):
    # take rt number and return virality class
    for i, t in enumerate(th):
        if rt <= t:
            return i
    return len(th)

tweets_df["virality"] = tweets_df["retweets"].apply(vir_transform)
counts, bins = np.histogram(tweets_df.virality, bins=len(RT_THRESHOLDS)+1)
fig = px.bar(x=[str(i) for i in range(len(RT_THRESHOLDS) + 1)], y=counts, labels={"x": "Virality", "y": "Count"},)
fig.show()
class Loader():
    def __init__(self, features, labels, batch_size=32, target_size=len(RT_THRESHOLDS) + 1):
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.batches = len(self.labels) // self.batch_size
        self.index = -1
        self.target_size = target_size
    
    def __len__(self):
        return self.batches
    
    def __iter__(self):
        self.index = -1
        return self
    
    def _ohe(self, lbls):
        ohe = torch.zeros((lbls.size, self.target_size))
        ohe[torch.arange(lbls.size), lbls.values] = 1
        return ohe

    def __next__(self):
        self.index += 1
        if self.index > self.batches:
            raise StopIteration
        start = self.index * self.batch_size 
        end = (self.index + 1) * self.batch_size
        return self.features[start:end], self._ohe(self.labels[start:end])
    from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(
    scaler.fit_transform(tweets_df[["hashtags", "mentions", "verified", "followers", "following", "text_len"]]),
    tweets_df[["virality"]],
    random_state=42,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
def fit(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, np.ravel(y_train))
    y_pred = model.predict(X_test)
    return scores(y_test, y_pred)
fit(LogisticRegression(solver="newton-cg"), X_train, y_train, X_test, y_test)
fit(SGDClassifier(), X_train, y_train, X_test, y_test)
fit(tree.DecisionTreeClassifier(), X_train, y_train, X_test, y_test)
fit(RandomForestClassifier(random_state=0), X_train, y_train, X_test, y_test)
tweets_df.info()
input_feats = ["text", "hashtags", "mentions", "followers", "following", "verified", "text_len", ]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = tweets_df.loc[:, input_feats].dropna()
df["virality"] = le.fit_transform(tweets_df["virality"])
df = df.sample(frac=1, random_state=42)
df.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[input_feats[4:]] = scaler.fit_transform(df[input_feats[4:]])
df.head()
(train_features, train_labels,
 val_features, val_labels,
 test_features, test_labels) = train_valid_test_split(df, target = 'virality', train_size=0.8, valid_size=0.1, test_size=0.1, random_state=42)
train_loader = Loader(train_features, train_labels)
val_loader = Loader(val_features, val_labels)
test_loader = Loader(test_features, test_labels)
def encode(df, tokenizer=tokenizer, inp="text"):
    text = df[inp]
    inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=False,
        )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

w = [len(train_labels[train_labels == i])/len(train_labels) for i in range(len(RT_THRESHOLDS) + 1)]
weights = torch.tensor(w, dtype=torch.float).to(device)
# From https://github.com/vandit15/Class-balanced-loss-pytorch.git
def focal_loss(logits, labels, alpha=weights, gamma=2.0):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

criterion = focal_loss
sm = nn.Softmax()
def train(model, arg_function, epoch, print_every=500, inp="text"):
    model.train()
    losses = []
    for i, (inputs, targets) in enumerate(train_loader):
        encode_df = inputs.loc[:, [inp]].apply(encode, inp=inp, axis=1, result_type="expand")
        ids = torch.stack(tuple(encode_df[0].values), 0).to(device)
        masks = torch.stack(tuple(encode_df[1].values), 0).to(device)
        kwargs = arg_function(inputs)
        outputs = model(ids, masks, **kwargs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets.to(device))
        losses.append(loss.item())
        if i % print_every == 0:
            print(f'Epoch: {epoch}, i: {i}, Loss:  {torch.mean(torch.tensor(loss)).item()}')
            losses = []
        loss.backward()
        optimizer.step()
        def validate(model, arg_function, inp="text"):
    model.eval()
    total_loss = []
    all_y = []
    all_y_pred = []
    for x, y in val_loader:
        encode_df = x.loc[:, [inp]].apply(encode, inp=inp, axis=1, result_type="expand")
        ids = torch.stack(tuple(encode_df[0].values), 0).to(device)
        masks = torch.stack(tuple(encode_df[1].values), 0).to(device)
        kwargs = arg_function(x)
        y_pred = model(ids, masks, **kwargs)
        loss = criterion(y_pred, y.to(device))
        y_pred = sm(y_pred)
        y_pred = torch.zeros(y_pred.shape).to(device).scatter(1, y_pred.argmax(1).unsqueeze(1), 1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        all_y.append(y.argmax(1))
        all_y_pred.append(y_pred.argmax(1))
        total_loss.append(loss.item())
    print("validation loss: ", torch.mean(torch.tensor(total_loss)).item())
    all_y = np.concatenate(all_y)
    all_y_pred = np.concatenate(all_y_pred)
    s = scores(all_y, all_y_pred)
    print(s)
    print(y)
    print(y_pred)
    def test(model, arg_function, inp="text"):
    model.eval()
    all_y = []
    all_y_pred = []
    for x, y in test_loader:
        encode_df = x.loc[:, [inp]].apply(encode, inp=inp, axis=1, result_type="expand")
        ids = torch.stack(tuple(encode_df[0].values), 0).to(device)
        masks = torch.stack(tuple(encode_df[1].values), 0).to(device)
        kwargs = arg_function(x)
        y_pred = model(ids, masks, **kwargs)
        y_pred = sm(y_pred)
        y_pred = torch.zeros(y_pred.shape).to(device).scatter(1, y_pred.argmax(1).unsqueeze(1), 1).cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        all_y.append(y.argmax(1))
        all_y_pred.append(y_pred.argmax(1))
    all_y = np.concatenate(all_y)
    all_y_pred = np.concatenate(all_y_pred)
    s = scores(all_y, all_y_pred)
    print(s)

    #+++++++++
    input_feats = ["text", "hashtags", "mentions", "followers", "following", "verified", "text_len", ]
    df = tweets_df.loc[:, input_feats].dropna()
df.head()

def combine_feats(x):
    return ". ".join([str(val) for val in x.values])
df["input"] = df.apply(combine_feats, axis=1)
df["virality"] = le.fit_transform(tweets_df["virality"])
(train_features, train_labels,
 val_features, val_labels,
 test_features, test_labels) = train_valid_test_split(df, target = 'virality', train_size=0.8, valid_size=0.1, test_size=0.1, random_state=42)
train_loader = Loader(train_features, train_labels)
val_loader = Loader(val_features, val_labels)
test_loader = Loader(test_features, test_labels)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
bert = AutoModel.from_pretrained("vinai/bertweet-base")

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True, normalization=True)
class Arch(nn.Module):
    def __init__(self, bert, sentiment_model):
        super(Arch, self).__init__()
        self.bert = bert
        self.sent = sentiment_model
        self.pre_classifier = nn.Linear(771, 771)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(771, 4)

    def forward(self, input_ids, attention_mask, sent_input_ids, sent_attention_mask):
        output_1 = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        sent_output = self.sent(input_ids=sent_input_ids, attention_mask=sent_attention_mask)
        pooler = torch.cat((hidden_state[:, 0], sent_output[0]), 1)
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    model = Arch(bert, sent_model).to(device)
optimizer = transformers.AdamW(params=model.parameters(), lr=1e-5)
def arg_func(x):
    clean_text = [preprocess(t) for t in x["text"].values]
    encoded = sent_tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    sent_input_ids = encoded["input_ids"].to(device)
    sent_attention_mask = encoded["attention_mask"].to(device)

    return {"sent_input_ids": sent_input_ids, "sent_attention_mask": sent_attention_mask}
for epoch in range(3):
    train(model, arg_func, epoch, inp="input")
    validate(model, arg_func, inp="input")
torch.save(model.state_dict(), "./models/vb.pt")
model = Arch(bert, sent_model).to(device)
model.load_state_dict(torch.load("./models/vb.pt"))
test(model, arg_func, inp="input")