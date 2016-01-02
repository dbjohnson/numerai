from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn import metrics


df = pd.DataFrame.from_csv('numerai_training_data.csv', index_col=None)
float_cols = [c for c in df.columns if c.startswith('f')]
#df['target'] = df['target'].map(lambda x: 1 if x else -1)
df_train = df[df['validation'] == 0]
df_test = df[df['validation'] == 1]

df_tourney = pd.DataFrame.from_csv('numerai_tournament_data.csv', index_col=None)


def area_under_curve(model, use_trainset=False):
    df_eval = df_train if use_trainset else df_test
    scores = model.predict(df_eval)
    uniq = np.unique(np.append(scores, [0, 1]))
    uniq.sort()
    tprs = []
    fprs = []

    for t in reversed(uniq):
        tn = len(df_eval[(scores < t) & (df_eval['target'] != 1)])
        tp = len(df_eval[(scores >= t) & (df_eval['target'] == 1)])
        fn = len(df_eval[(scores < t) & (df_eval['target'] == 1)])
        fp = len(df_eval[(scores >= t) & (df_eval['target'] != 1)])
        tpr = float(tp) / max(1, tp + fn)
        fpr = float(fp) / max(1, tn + fp)
        tprs.append(tpr)
        fprs.append(fpr)

    pos_idx = (df_eval['target'] == 1).values
    neg_idx = ~pos_idx
    plt.subplot(121)
    plt.hist(scores[neg_idx], 50, color='r', alpha=0.5)
    plt.hist(scores[pos_idx], 50, color='b', alpha=0.5)
    plt.subplot(122)
    plt.plot(fprs, tprs)
    plt.title(auc)
    plt.plot([0, 1], [0, 1])
    print metrics.auc(fprs, tprs)


def make_submission(model, fn='submission.csv'):
    scores = model.predict(df_tourney)
    df_tourney['probability'] = scores
    df_tourney[['t_id', 'probability']].to_csv(fn)
    print 'Submission complete!'


def make_cv_set(cv=4, seed=1337):
    np.random.seed(seed)

    cv_to_train_dfcs = defaultdict(list)
    cv_to_test_dfcs = defaultdict(list)
    for x, dfx in df.groupby('c1'):
        cv_sets = np.random.choice(cv, size=len(dfx))
        for c in xrange(cv):
            train = dfx[cv_sets != c]
            test = dfx[cv_sets == c]
            cv_to_train_dfcs[c].append(train)
            cv_to_test_dfcs[c].append(test)

    train_validation_sets = []
    for c in xrange(cv):
        train = pd.concat(cv_to_train_dfcs[c])
        test = pd.concat(cv_to_test_dfcs[c])
        train_validation_sets.append((train, test))

    return train_validation_sets
