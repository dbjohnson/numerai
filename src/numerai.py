import numpy as np
from matplotlib import pylab as plt

import models
import scoring


cross_validation_sets = scoring.make_cv_set(cv=4)


def explore(pc_range=xrange(1, len(scoring.float_cols)+1), cluster_range=xrange(1, 10), discard=0.05):
    reload(models)
    best_score = None
    min_aucs = []
    max_aucs = []
    for n_princ_comp in pc_range:
        min_aucs.append([])
        max_aucs.append([])
        for n_clusters in cluster_range:
            m, min_auc, max_auc = make_model(n_princ_comp=n_princ_comp, n_clusters=n_clusters, discard=discard)
            print 'pc: {}, clusters: {}, discard: {}, min auc: {} max auc: {}'.format(n_princ_comp, n_clusters, discard, min_auc, max_auc)
            min_aucs[-1].append(min_auc)
            max_aucs[-1].append(max_auc)
            if not best_score or min_auc > best_score:
                print 'new best!'
                best_score = min_auc
                # retrain model on full training set for submission
                fit_model(m, scoring.df, discard)
                scoring.make_submission(m, '_min{:1.5f}_max{:1.4f}_p{}_k{}_d{}.csv'.format(min_auc, max_auc, n_princ_comp, n_clusters, discard))

    plt.subplot(121)
    plt.imshow(min_aucs, interpolation='nearest')
    plt.subplot(122)
    plt.imshow(max_aucs, interpolation='nearest')


def make_model(n_princ_comp, n_clusters, discard):
    m = models.MixedModelKmeans(n_clusters=n_clusters, n_princ_comp=n_princ_comp)
    scores = []
    for df_train, df_validation in cross_validation_sets:
        fit_model(m, df_train, discard)
        scores.append(m.score(df_validation))
    return m, min(scores), max(scores)


def fit_model(m, df, discard):
    m.fit(df)
    if discard > 0:
        residuals = np.abs(df['target'] - m.predict(df))
        thresh = residuals.quantile(1 - discard)
        m.fit(df.drop(df.index[np.where(residuals > thresh)]))

