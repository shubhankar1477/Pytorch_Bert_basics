import pandas as pd
from sklearn.model_selection import StratifiedKFold
def create_folds():
    df = pd.read_csv("IMDB Dataset.csv")
    df['target']=df['sentiment'].apply(lambda x:1 if x=='positive' else 0)
    df['kfold']=-1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target
    kf = StratifiedKFold(n_splits=5)
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f
    df.to_csv("./imdb_folds.csv")

if __name__ == "__main__":
    create_folds()