## Ham_spam_email_classification
This repository contains code for the classification of ham and spam emails on subject lines using (1) PCA used with Logistic regression (2) CNN deep learning.

## Dataset
The dataset is based on cleaned Enron corpus, there are a total of 92188 messages belonging to 158 users with an average of 757 messages per user. The dataset has almost an equal distribution of ham and spam emails. In this study 2000 emails are used 1000 ham + 1000 spams. The data are split into 1600 examples for training, 400 for validation and 58000 for testing.

```
from ipynb.fs.full.dataset import spam_test_train_set
[train_email, test_email] = spam_test_train_set()
```
## Saved models

### PCA model
```
filename = 'PCA_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pca = PCA(n_components=160)
topic_vectors = pca.fit_transform(X)
loaded_model.score(topic_vectors, Y)
```
### CNN model
```
from keras.models import model_from_json
json_file = open('model1000.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1000.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
