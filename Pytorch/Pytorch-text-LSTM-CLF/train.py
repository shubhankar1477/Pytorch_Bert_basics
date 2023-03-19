import io
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
import config
import dataset
import engine
import lstm
from tqdm import tqdm
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    print("Progress OF Loadinng vectors")
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def create_embedding_matrix(word_index,embedding_dict):
    embedding_matrix = np.zeros((len(word_index)+1,300))
    print("Progress of creating matrix")
    for word , i in tqdm(word_index.items()):
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix


def run(df,fold):
    train_df = df[df.kfold!=fold].reset_index(drop=True)
    valid_df = df[df.kfold==fold].reset_index(drop=True)

    print("fitting Tokenizer")
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    xtrain = tokenizer.texts_to_sequences(train_df.review.values.tolist())            
    xtest = tokenizer.texts_to_sequences(valid_df.review.values.tolist())            
    # padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain,maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest,maxlen=config.MAX_LEN)

    ## Initilase dataset
    train_dataset = dataset.IMDBDataset(reviews=xtrain,targets=train_df.target.values)
    valid_dataset = dataset.IMDBDataset(reviews=xtest,targets=valid_df.target.values)


    ## Initialise dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,batch_size=config.TRAIN_BATCH_SIZE
    )

    test_data_loader = torch.utils.data.DataLoader(
        valid_dataset,batch_size=config.VALID_BATCH_SIZE
    )

    print("Load Embedding")

    embedding_dict = load_vectors("crawl-300d-2M.vec")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index,embedding_dict)

    device = torch.device("cpu")

    model = lstm.LSTM(embedding_matrix)

    model.to(device=device)

    optimiser = torch.optim.Adam(model.parameters(),lr=1e-3)

    print("Start Training...")

    best_accuracy = 0
    early_stopping_counter= 0

    for epoch in range(config.EPOCHS):
        engine.train(train_data_loader,model,optimiser=optimiser,device=device
                     )
        outputs,targets = engine.evaluate(test_data_loader,model,device)

        outputs = np.array(outputs) >=0.5

        accuracy = metrics.accuracy_score(targets,outputs)

        print(f"FOLD {fold} ,EPOCH {epoch} , Accuracy {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter +=1
        if early_stopping_counter > 2:
            break

if __name__ == "__main__":
    df = pd.read_csv("/Users/shubhankardeshpande/Desktop/Projects/Pytroch/Pytroch-text-classification/imdb_folds.csv")
    run(df,fold = 0)









