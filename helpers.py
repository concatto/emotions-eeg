from collections import ChainMap
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.utils import check_random_state, shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import pickle
from copy import deepcopy


# To be merged in https://github.com/scikit-learn/scikit-learn/issues/13621
class RepeatedStratifiedGroupKFold():

    def __init__(self, n_splits=5, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    # Implementation based on this kaggle kernel:
    #    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    def split(self, X, y=None, groups=None):
        k = self.n_splits

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std(
                    [y_counts_per_fold[i][label] / y_distr[label]
                        for i in range(k)]
                )
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        rnd = check_random_state(self.random_state)
        for repeat in range(self.n_repeats):
            labels_num = np.max(y) + 1
            y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
            y_distr = Counter()
            for label, g in zip(y, groups):
                y_counts_per_group[g][label] += 1
                y_distr[label] += 1

            y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
            groups_per_fold = defaultdict(set)

            groups_and_y_counts = list(y_counts_per_group.items())
            rnd.shuffle(groups_and_y_counts)

            for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
                best_fold = None
                min_eval = None
                for i in range(k):
                    fold_eval = eval_y_counts_per_fold(y_counts, i)
                    if min_eval is None or fold_eval < min_eval:
                        min_eval = fold_eval
                        best_fold = i
                y_counts_per_fold[best_fold] += y_counts
                groups_per_fold[best_fold].add(g)

            all_groups = set(groups)
            for i in range(k):
                train_groups = all_groups - groups_per_fold[i]
                test_groups = groups_per_fold[i]

                train_indices = [i for i, g in enumerate(
                    groups) if g in train_groups]
                test_indices = [i for i, g in enumerate(
                    groups) if g in test_groups]

                yield train_indices, test_indices
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, groups=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       groups=groups,
                       train_sizes=train_sizes,
                       return_times=True,
                       shuffle=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        print(params.keys())
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = list(models.keys())
        self.grid_searches = {}
        self.cross_val_results = {}
        self.ran_with_outer_cv = False

    @staticmethod
    def load(persist_dir):
        with open(persist_dir, 'rb') as f:
            instance = pickle.load(f)
        return instance

    def save(self, persist_dir):
        with open(persist_dir, 'wb') as f:
            pickle.dump(self, f)
            print("Object persisted")

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False, groups=None, outer_cv=None, persist_dir=None, randomSearchFor=None):
        for key in self.keys:
            model = self.models[key]
            params = self.params[key]

            if randomSearchFor != None and key in randomSearchFor:
                SearchCV = RandomizedSearchCV
            else:
                SearchCV = GridSearchCV

            gs = SearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=True,
                              return_train_score=True)

            if outer_cv != None:
                print("Running GridSearchCV for %s with nested cross validation." % key)
                self.cross_val_results[key] = cross_validate(gs, X, y, groups=groups, cv=outer_cv, scoring=scoring, error_score='raise',
                                                             return_train_score=True, return_estimator=True, verbose=10, fit_params={'groups': groups})
                self.ran_with_outer_cv = True
                gs = max(
                    self.cross_val_results[key]['estimator'], key=lambda item: item.best_score_)

            else:
                self.ran_with_outer_cv = False
                print("Running GridSearchCV for %s." % key)
                gs.fit(X, y, groups=groups)

            self.grid_searches[key] = gs

            if persist_dir != None:
                self.save(persist_dir)
            # print("Score: ", gs.best_score_)

    def score_summary(self, sort_by='mean_score'):

        if self.ran_with_outer_cv:
            rows = []

            

            for k in self.keys:
                cv_results_params = deepcopy(self.grid_searches[k].cv_results_['params'])

                # convert list values into string, e.g. (1, 2) = > '(1,2)'
                for dictionary in cv_results_params:
                    for key in dictionary:
                        dictionary[key] = str(dictionary[key])

                test_scores = self.cross_val_results[k]['test_score']
                train_scores = self.cross_val_results[k]['train_score']
                row = pd.DataFrame({
                    'estimator': [k],
                    'min_score': test_scores.min(),
                    'max_score': test_scores.max(),
                    'mean_score': test_scores.mean(),
                    'std_score': test_scores.std(),
                    'train_mean_score': train_scores.mean(),
                    **dict(ChainMap(*cv_results_params)),
                })

                rows.append(row)

            df = pd.concat(rows)

            return df
        else:

            def row(key, scores, params):
                d = {
                    'estimator': key,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                }
                return pd.Series({**params, **d})

            rows = []

            for k in self.grid_searches:

                # [{'n_estimators': 16}, {'n_estimators': 32}]
                params = self.grid_searches[k].cv_results_['params']
                scores = []
                for i in range(self.grid_searches[k].n_splits_):
                    key = "split{}_test_score".format(i)
                    r = self.grid_searches[k].cv_results_[key]  # [0.96694215 0.95041322]
                    scores.append(r.reshape(len(params), 1))

                all_scores = np.hstack(scores)
                for p, s in zip(params, all_scores):
                    rows.append((row(k, s, p)))

            df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

            columns = ['estimator', 'min_score','mean_score', 'max_score', 'std_score']
            columns = columns + [c for c in df.columns if c not in columns]

            return df[columns]
