# Dataset, splits, generators, evaluator and abstract model handlers

from sklearn.manifold import TSNE
import matplotlib
import bottleneck as bn
from random import randrange, shuffle
import random
import tensorflow_addons as tfa
import tensorflow as tf
from time import time
import numpy as np
import pandas as pd
import argparse
import os

DEFAULT_SEED = 42

SEED = DEFAULT_SEED
# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# Auxiliary functions

def set_seed(seed=DEFAULT_SEED):
    """
    Set random seed in all used libs.
    """
    global SEED
    SEED = seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def get_seed():
    return SEED


def shufflestr(x):
    """
    Shuffle randomly items in string separated by comma.
    """
    p = x.split(',')
    random.shuffle(p)
    return ",".join(p)


def split2_50(x):
    """
    Returns first half of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .5)
    return ",".join(p[:s])


def split1_50(x):
    """
    Returns second half of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .5)
    return ",".join(p[s:])


def split75(x):
    """
    Returns first three quarters of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .75)
    return ",".join(p[:s])


def split25(x):
    """
    Returns last quarter of items in string separated by comma.
    """
    p = x.split(',')
    s = int(len(p) * .75)
    return ",".join(p[s:])


# TF functions

@tf.function
def cosmse(x, y):
    # x=tf.cast(x,'float32')
    # y=tf.cast(y,'float32')
    a = tf.constant(ALPHA) * cosine_loss(x, y)
    d = tf.constant(BETA) * tf.keras.losses.MSE(x, y)
    return a + d


@tf.function
def cosine_loss(x, y):
    return tf.keras.losses.cosine_similarity(x, y) + tf.constant(1.0)


# Dataset classes
class Data:
    """
    Basic dataset class.
    """

    def __init__(self, d=str(), pruning=False):
        self.directory = d
        if pruning:
            self.read_all(p="_p" + pruning)
        else:
            self.read_all()
        # array of all data splits
        self.splits = []
        # direct link to default split
        self.split = None

    def train_tokenizer(self):
        self.toki = tf.keras.preprocessing.text.Tokenizer()
        # bz_toki = rec.Itemizer()
        # bz_toki.fit_on_texts(stats.itemid.to_list())
        self.toki.fit_on_texts(self.items_sorted.itemid.to_list())
        _, self.num_words = self.toki.texts_to_matrix(['xx']).shape
        print("Tokenizer trained for", self.num_words, "items.")

    def create_splits(self, n, k_test, shuffle=True, n_fold=True, generators=True, batch_size=1024):
        """
        Create n splits of k test items
        shuffle = shuffle users on begin
        n_fold = create n disjunct folds of data
        """
        if not len(self.splits) == 0:
            print("Splits are not empty! Doing nothing...")
            return

        print("Creating", n, "splits of", k_test, "samples each.")
        if shuffle:
            print("Initial user shuffle.")
            self.users = self.users.sample(frac=1, random_state=get_seed())

        for i in range(n):
            print("Creating split nr.", i + 1)
            self.splits.append(
                Split(self, k_test, shuffle=False, index=i * k_test, generator=generators, batch_size=batch_size))

        self.split = self.splits[0]

    def save_splits(self):
        for i in range(len(self.splits)):
            d = "split_" + str(i + 1)
            print(d)
            os.makedirs(d)
            self.splits[i].train_users.to_json(d + "/train_users.json")
            self.splits[i].test_users.to_json(d + "/test_users.json")

    def load_splits(self, split=0):
        if split == 0:
            for i in range(len(self.splits)):
                d = "split_" + str(i + 1)
                print(d)
                self.splits[i].train_users = pd.read_json(d + "/train_users.json").userid.apply(str).to_frame()
                self.splits[i].test_users = pd.read_json(d + "/test_users.json").userid.apply(str).to_frame()
                self.splits[i].generators()
        else:
            d = "split_" + str(split)
            print(d)
            self.splits[0].train_users = pd.read_json(d + "/train_users.json").userid.apply(str).to_frame()
            self.splits[0].test_users = pd.read_json(d + "/test_users.json").userid.apply(str).to_frame()
            self.splits[0].generators()

    def read_users(self, p=''):
        print("Reading users" + p)
        self.users = pd.read_json(self.directory + 'users' + p + '.json')
        self.users['userid'] = self.users.userid.apply(str)

    def read_items(self, p=''):
        print("Reading items" + p)
        self.items = pd.read_json(self.directory + 'items' + p + '.json')
        self.items['itemid'] = self.items.itemid.apply(str)

    def read_ratings(self, p=''):
        print("Reading ratings" + p)
        self.ratings = pd.read_json(self.directory + 'ratings' + p + '.json')
        self.ratings['userid'] = self.ratings.userid.apply(str)
        self.ratings['itemid'] = self.ratings.itemid.apply(str)

    def read_purchases(self, p=''):
        print("Reading purchases" + p)
        self.purchases = pd.read_json(self.directory + 'purchases' + p + '.json')
        self.purchases['userid'] = self.purchases.userid.apply(str)
        self.purchases['itemid'] = self.purchases.itemid.apply(str)

    def read_purchases_txt(self, p=''):
        print("Reading purchases_txt" + p)
        self.purchases_txt = pd.read_json(self.directory + 'purchases_txt' + p + '.json')
        self.purchases_txt['userid'] = self.purchases_txt.userid.apply(str)

    def read_items_sorted(self, p=''):
        print("Reading items_sorted" + p)
        self.items_sorted = pd.read_json(self.directory + 'items_sorted' + p + '.json')
        self.items_sorted['itemid'] = self.items_sorted.itemid.apply(str)

    def read_users_sorted(self, p=''):
        print("Reading users_sorted" + p)
        self.users_sorted = pd.read_json(self.directory + 'users_sorted' + p + '.json')
        self.users_sorted['userid'] = self.users_sorted.userid.apply(str)

    def read_all(self, p=''):
        now = time()
        self.read_users(p)
        self.read_items(p)
        # self.read_purchases(p)
        self.read_purchases_txt(p)
        self.read_items_sorted(p)
        self.read_users_sorted(p)
        print("Read all in", time() - now)
        self.train_tokenizer()


class Split:
    """
    Definition of train/validation/test subsets of the dataset.
    """

    def __init__(self, data, k_test, shuffle=True, index=0, generator=True, batch_size=1024):
        """
        Create split of k test items
        shuffle = shuffle users on begin
        n_fold = create n disjunct folds of data
        """
        self.master_data = data
        if shuffle:
            self.all_users = data.users.sample(frac=1).copy(deep=True)
        else:
            self.all_users = data.users.copy(deep=True)

        self.test_users = self.all_users.iloc[index:index + k_test]
        self.train_users = self.all_users[~self.all_users.userid.isin(self.test_users.userid)]
        self.validation_users = self.train_users.iloc[index:index + k_test]
        self.train_users = self.train_users[~self.train_users.userid.isin(self.validation_users.userid)]
        if generator:
            self.generators(batch_size=batch_size)

    def generators(self,
                   batch_size=1024,
                   random_batching=True,
                   prevent_identity=True,
                   full_data=False,
                   p50_splits=True,
                   p2575_splits=False,
                   p7525_splits=False,
                   p2525_splits=False,
                   p7575_splits=False
                   ):
        self.train_gen = SplitGenerator(
            data_df=self.train_purchases_txt(),
            itemizer=self.master_data.toki,
            batch_size=batch_size,
            random_batching=random_batching,
            prevent_identity=prevent_identity,
            full_data=full_data,
            p50_splits=p50_splits,
            p2575_splits=p2575_splits,
            p7525_splits=p7525_splits,
            p2525_splits=p2525_splits,
            p7575_splits=p7575_splits
        )

        self.test_gen = SplitGenerator(
            data_df=self.test_purchases_txt(),
            itemizer=self.master_data.toki,
            batch_size=128,
            random_batching=False,
            prevent_identity=False,
            full_data=True,
            p50_splits=False,
            p2575_splits=False,
            p7525_splits=False,
            p2525_splits=False,
            p7575_splits=False
        )

        self.validation_gen = SplitGenerator(
            data_df=self.validation_purchases_txt(),
            itemizer=self.master_data.toki,
            batch_size=128,
            random_batching=False,
            prevent_identity=False,
            full_data=True,
            p50_splits=False,
            p2575_splits=False,
            p7525_splits=False,
            p2525_splits=False,
            p7575_splits=False
        )
        print("Creating evaluator")

        np.random.seed(get_seed())
        random.seed(get_seed())
        self.test_evaluator = Evaluator(self, data="test")
        np.random.seed(get_seed())
        random.seed(get_seed())
        self.evaluator = Evaluator(self, data="val")

    def train_purchases_txt(self):
        return self.master_data.purchases_txt[self.master_data.purchases_txt.userid.isin(self.train_users.userid)].copy(
            deep=True)

    def test_purchases_txt(self):
        return self.master_data.purchases_txt[self.master_data.purchases_txt.userid.isin(self.test_users.userid)].copy(
            deep=True)

    def validation_purchases_txt(self):
        return self.master_data.purchases_txt[
            self.master_data.purchases_txt.userid.isin(self.validation_users.userid)].copy(deep=True)


class SplitGenerator(tf.keras.utils.Sequence):
    """
    TF data generator.
    """

    def __init__(
            self,
            data_df,
            itemizer,
            batch_size=128,
            random_batching=True,
            prevent_identity=False,
            full_data=True,
            p50_splits=True,
            p2575_splits=False,
            p7525_splits=False,
            p2525_splits=False,
            p7575_splits=False):

        now = time()
        self.prevent_identity = prevent_identity
        self.full_data = full_data
        self.p50_splits = p50_splits
        self.p2575_splits = p2575_splits
        self.p7525_splits = p7525_splits
        self.p2525_splits = p2525_splits
        self.p7575_splits = p7575_splits
        self.toki = itemizer
        self.batch_size = batch_size
        self.data = data_df
        # self.data_np = self.data.to_numpy()

        self.length = len(self.data)
        self.random_batching = random_batching
        self.on_epoch_end()
        print("SplitGenerator init done in", time() - now, "secs.")

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.length / self.batch_size)) - 1

    def __call__(self, batch_size):
        """Allows to use the size of batch when calling the training."""
        self.batch_size = batch_size
        return self

    def on_epoch_end(self):
        if self.random_batching:
            self.data = self.data.sample(frac=1)

        self.data['temp_itemids_p'] = self.data['itemids'].apply(shufflestr)
        self.data['temp_itemids_p1_50'] = self.data['temp_itemids_p'].apply(split1_50)
        self.data['temp_itemids_p2_50'] = self.data['temp_itemids_p'].apply(split2_50)
        self.data['temp_itemids_p_25'] = self.data['temp_itemids_p'].apply(split25)
        self.data['temp_itemids_p_75'] = self.data['temp_itemids_p'].apply(split75)
        self.data_np = self.data.to_numpy()

    def get_basket_np(self, items):
        if self.n_ratings:
            return np.vstack([self.embeddings_dict.get(x, self.null_val) for x in items.split(',')])

        return np.vstack([self.embeddings_dict.get(x, self.null_val) for x in set(items.split(','))])

    def __getitem__(self, index):
        # binary mode = output vectors is 0/1 only
        mod = 'binary'

        data_slice = self.data_np[self.batch_size * index:self.batch_size * index + self.batch_size]

        indices = list(range(self.__len__()))
        indices += indices

        index2 = indices[index + 1]
        index3 = indices[index + 2]
        index4 = indices[index + 3]
        index5 = indices[index + 4]

        if self.full_data:
            data_slice = self.data_np[self.batch_size * index:self.batch_size * index + self.batch_size]

        if self.p50_splits:
            data_slice2 = self.data_np[self.batch_size * index2:self.batch_size * index2 + self.batch_size]
            data_slice3 = self.data_np[self.batch_size * index3:self.batch_size * index3 + self.batch_size]

        if self.p2575_splits or self.p7525_splits or self.p2525_splits or self.p7575_splits:
            data_slice4 = self.data_np[self.batch_size * index4:self.batch_size * index4 + self.batch_size]
            data_slice5 = self.data_np[self.batch_size * index5:self.batch_size * index5 + self.batch_size]

        ret_x = []
        ret_y = []

        # full input to full_output
        if self.full_data:
            ret_x.append(self.toki.texts_to_matrix(data_slice[:, 1], mode=mod))
            ret_y.append(self.toki.texts_to_matrix(data_slice[:, 1], mode=mod))

        if self.p50_splits:
            ret_x.append(self.toki.texts_to_matrix(data_slice2[:, 3], mode=mod))
            ret_x.append(self.toki.texts_to_matrix(data_slice3[:, 4], mode=mod))

            if self.prevent_identity:
                ret_y.append(self.toki.texts_to_matrix(data_slice2[:, 4], mode=mod))
                ret_y.append(self.toki.texts_to_matrix(data_slice3[:, 3], mode=mod))
            else:
                ret_y.append(self.toki.texts_to_matrix(data_slice2[:, 3], mode=mod))
                ret_y.append(self.toki.texts_to_matrix(data_slice3[:, 4], mode=mod))

        if self.p2575_splits:
            ret_x.append(self.toki.texts_to_matrix(data_slice4[:, 5], mode=mod))
            ret_y.append(self.toki.texts_to_matrix(data_slice4[:, 6], mode=mod))

        if self.p7525_splits:
            ret_x.append(self.toki.texts_to_matrix(data_slice4[:, 6], mode=mod))
            ret_y.append(self.toki.texts_to_matrix(data_slice4[:, 5], mode=mod))

        if self.p2525_splits:
            ret_x.append(self.toki.texts_to_matrix(data_slice5[:, 5], mode=mod))
            ret_y.append(self.toki.texts_to_matrix(data_slice5[:, 5], mode=mod))

        if self.p7575_splits:
            ret_x.append(self.toki.texts_to_matrix(data_slice5[:, 6], mode=mod))
            ret_y.append(self.toki.texts_to_matrix(data_slice5[:, 6], mode=mod))

        return np.vstack(ret_x), np.vstack(ret_y)


class Evaluator:
    """
    Evaluation on give data split.
    """

    def __init__(self, split, method='leave_random_20_pct_out', data="test", debug=False):
        assert method in ['leave_random_20_pct_out', '1_20', '2_20', '3_20', '4_20', '5_20']
        self.split = split
        if data == "val":
            print("Creating validation split evaluator with", method, "method.")
            self.ivx = split.validation_gen.data.set_index(split.validation_gen.data.userid).sort_index().itemids.apply(
                lambda x: x.split(','))
        else:
            print("Creating test split evaluator with", method, "method.")
            self.ivx = split.test_gen.data.set_index(split.test_gen.data.userid).sort_index().itemids.apply(
                lambda x: x.split(','))
        if debug:
            print("Stage 1 done.")
        self.tpx = []
        if method == 'leave_random_20_pct_out':
            for e in range(len(self.ivx)):
                tech20 = []
                num_to_add = int(len(self.ivx[e]) * 0.2)
                if num_to_add < 1:
                    num_to_add = 1
                for x in range(num_to_add):
                    random_pick_index = randrange(len(self.ivx[e]))
                    tech20.append(self.ivx[e].pop(random_pick_index))
                self.tpx.append(tech20)
        else:
            end = int(method[0])
            start = end - 1
            random.seed(get_seed())
            for e in range(len(self.ivx)):
                tech20 = []
                interactions_len = len(self.ivx[e])
                num_to_add = int(interactions_len * 0.2)
                if num_to_add < 1:
                    num_to_add = 1
                shuffle(self.ivx[e])
                for x in range(start * num_to_add, end * num_to_add):
                    random_pick_index = x
                    if random_pick_index >= len(self.ivx[e]):
                        random_pick_index = len(self.ivx[e]) - 1
                    tech20.append(self.ivx[e].pop(random_pick_index))
                self.tpx.append(tech20)

        if debug:
            print("Stage 2 done.")
        self.iv = self.split.master_data.toki.texts_to_matrix(
            [",".join(x) for x in self.ivx]
        )
        if debug:
            print("Stage 3 done.")
        self.tp = self.split.master_data.toki.texts_to_matrix(
            [",".join(x) for x in self.tpx]
        )
        self.tpx_set = [set(t) for t in self.tpx]
        if debug:
            print("Stage 4 done.")

    def update(self, m, chunk=1000):
        assert len(self.iv) % chunk == 0
        self.pr = np.vstack([m.predict(self.iv[chunk * x:chunk * (x + 1)]) for x in range(len(self.iv) // chunk)])
        self.ppp = (1 - self.iv) * self.pr
        self.ppp[:, 0] = 0

    def get_ncdg(self, k):
        pr = self.pr
        ppp = self.ppp
        idx = bn.argpartition(-ppp, k, axis=1)
        topk_part = ppp[np.arange(ppp.shape[0])[:, np.newaxis], idx[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx[np.arange(self.iv.shape[0])[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k + 2))
        z = zip([[self.split.master_data.toki.index_word[b] for b in a] for a in idx_topk], self.tpx_set)
        n = np.array([(np.array([1 if x in true else 0 for x in pred]) * tp).sum() for pred, true in z])
        d = np.array([(np.ones(min(k, len(x))) * tp[:len(x)]).sum() for x in self.tpx_set])
        return (n / d).mean()

    def get_hr(self, k):
        pr = self.pr
        ppp = self.ppp
        idx = bn.argpartition(-ppp, k, axis=1)
        z = zip([set([self.split.master_data.toki.index_word[s] for s in a[:k]]) for a in idx], self.tpx_set)
        r = [0 if len(pred & true) / min(k, len(true)) == 0 else 1 for pred, true in z]
        return sum(r) / len(r)

    def get_coverage(self, k):
        pr = self.pr
        ppp = self.ppp

        idx = bn.argpartition(-ppp, k, axis=1)
        covered = len(set().union(*[set([self.split.master_data.toki.index_word[s] for s in a[:k]]) for a in idx]))
        total = self.iv.shape[1] - 1
        return covered / total

    def get_recall(self, k):
        pr = self.pr
        ppp = self.ppp
        idx = bn.argpartition(-ppp, k, axis=1)
        z = zip([set([self.split.master_data.toki.index_word[s] for s in a[:k]]) for a in idx], self.tpx_set)
        r = [len(pred & true) / min(k, len(true)) for pred, true in z]
        return sum(r) / len(r)

    def ncdg(self, m, k):
        c = getattr(m, "pred_from_mean", None)
        if not callable(c):
            c = getattr(m, "predict", None)
        pr = c(self.iv)
        ppp = (1 - self.iv) * pr
        ppp[:, 0] = 0
        idx = bn.argpartition(-ppp, k, axis=1)
        topk_part = ppp[np.arange(ppp.shape[0])[:, np.newaxis], idx[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx[np.arange(self.iv.shape[0])[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k + 2))
        z = zip([[self.split.master_data.toki.index_word[b] for b in a] for a in idx_topk], self.tpx_set)
        n = np.array([(np.array([1 if x in true else 0 for x in pred]) * tp).sum() for pred, true in z])
        d = np.array([(np.ones(min(k, len(x))) * tp[:len(x)]).sum() for x in self.tpx_set])
        return (n / d).mean()

    def hr(self, m, k):
        c = getattr(m, "pred_from_mean", None)
        if not callable(c):
            c = getattr(m, "predict", None)
        pr = c(self.iv)
        ppp = (1 - self.iv) * pr
        ppp[:, 0] = 0
        idx = bn.argpartition(-ppp, k, axis=1)
        z = zip([set([self.split.master_data.toki.index_word[s] for s in a[:k]]) for a in idx], self.tpx_set)
        r = [0 if len(pred & true) / min(k, len(true)) == 0 else 1 for pred, true in z]
        return sum(r) / len(r)

    def coverage(self, m, k):
        c = getattr(m, "pred_from_mean", None)
        if not callable(c):
            c = getattr(m, "predict", None)
        pr = c(self.iv)
        ppp = (1 - self.iv) * pr
        ppp[:, 0] = 0
        idx = bn.argpartition(-ppp, k, axis=1)
        covered = len(set().union(*[set([self.split.master_data.toki.index_word[s] for s in a[:k]]) for a in idx]))
        total = self.iv.shape[1] - 1
        return covered / total

    def recall(self, m, k):
        c = getattr(m, "pred_from_mean", None)
        if not callable(c):
            c = getattr(m, "predict", None)
        pr = c(self.iv)
        ppp = (1 - self.iv) * pr
        ppp[:, 0] = 0
        idx = bn.argpartition(-ppp, k, axis=1)
        z = zip([set([self.split.master_data.toki.index_word[s] for s in a[:k]]) for a in idx], self.tpx_set)
        r = [len(pred & true) / min(k, len(true)) for pred, true in z]
        return sum(r) / len(r)


# Model abstract class
class Model:
    """
    Abstract model.

    Subclassed model should implements create_model and train_model.

    """

    def __init__(self, split, name):
        self.split = split
        self.dataset = split.master_data
        self.metrics = {
            'Recall@5': {'k': 5, 'method': self.split.evaluator.get_recall, 'value': None},
            'Recall@20': {'k': 20, 'method': self.split.evaluator.get_recall, 'value': None},
            'Recall@50': {'k': 50, 'method': self.split.evaluator.get_recall, 'value': None},
            'NCDG@100': {'k': 100, 'method': self.split.evaluator.get_ncdg, 'value': None},
            'Coverage@5': {'k': 5, 'method': self.split.evaluator.get_coverage, 'value': None},
            'Coverage@20': {'k': 20, 'method': self.split.evaluator.get_coverage, 'value': None},
            'Coverage@50': {'k': 50, 'method': self.split.evaluator.get_coverage, 'value': None},
            'Coverage@100': {'k': 100, 'method': self.split.evaluator.get_coverage, 'value': None},
        }
        self.name = name

    def create_model(self):
        """
        Build Your own model here
        """

    def train_model(self):
        """
        Create your own training loop here
        """

    def evaluate_model(self):
        self.split.evaluator.update(self.model)
        for x in self.metrics.values():
            x['value'] = x['method'](x['k'])

    def print_metrics(self):
        print("Model metrics:", end='')
        for k, x in self.metrics.items():
            print(k, end="=")
            print(round(x['value'], 4), end=" ")
        print()

    def test_model(self):
        e = self.split.test_evaluator
        e.update(self.model)
        print("Results for test set: Recall@20=", e.get_recall(20), ", Recall@50=", e.get_recall(50), ", NCDG@100=",
              e.get_ncdg(100), sep="")
        with open("seed_results_test.txt", "a") as myfile:
            myfile.write("Results for test set: Recall@20=" + str(e.get_recall(20)) + ", Recall@50=" + str(
                e.get_recall(50)) + ", NCDG@100=" + str(e.get_ncdg(100)) + "\n")

    def test_model_val(self):
        e = self.split.evaluator
        e.update(self.model)
        print("Results for validation set: Recall@20=", e.get_recall(20), ", Recall@50=", e.get_recall(50),
              ", NCDG@100=",
              e.get_ncdg(100), sep="")
        with open("seed_results_val.txt", "a") as myfile:
            myfile.write("Results for validation set: Recall@20=" + str(e.get_recall(20)) + ", Recall@50=" + str(
                e.get_recall(50)) + ", NCDG@100=" + str(e.get_ncdg(100)) + "\n")


# Tensorflow objects - Callbacks
class MetricsCallback(tf.keras.callbacks.Callback):
    """
    Evaluate model in tf callback.
    """

    def __init__(self, rsmodel):
        super(MetricsCallback, self).__init__()
        self.epoch = 0
        self.loss_metrics = dict()
        self.eval_metrics = dict()
        self.evaluate_loss_metrics = ['loss', 'val_loss']
        self.rsmodel = rsmodel
        self.best_ncdg100 = 0.
        self.best_ncdg100_epoch = 0
        self.best_recall20 = 0.
        self.best_recall20_epoch = 0
        self.best_recall50 = 0.
        self.best_recall50_epoch = 0
        self.tsne_df = pd.DataFrame(columns=["epoch", "tsne_coords"])

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1

        self.loss_metrics[self.epoch] = dict()
        self.eval_metrics[self.epoch] = dict()
        # add metrics from logs
        for x in self.evaluate_loss_metrics:
            self.loss_metrics[self.epoch][x] = logs[x]
        # add custom metrics
        self.rsmodel.evaluate_model()
        self.rsmodel.print_metrics()
        for x in self.rsmodel.metrics.keys():
            self.eval_metrics[self.epoch][x] = self.rsmodel.metrics[x]['value']
        self.ncdg_100_watch()
        self.recall20_watch()
        self.recall50_watch()
        # self.get_history_df()
        # self.calc_tsne()

    def recall20_watch(self):
        if self.eval_metrics[self.epoch]['Recall@20'] > self.best_recall20:
            print("New best for Recall@20")
            self.model.save_weights(self.rsmodel.name + "_best_recall_20/" + self.rsmodel.name)
            self.best_recall20 = self.eval_metrics[self.epoch]['Recall@20']
            self.best_recall20_epoch = self.epoch
            with open(self.rsmodel.name + "_best_recall_20/" + "epoch.txt", "w") as text_file:
                text_file.write(str(self.best_recall20_epoch))

    def recall50_watch(self):
        if self.eval_metrics[self.epoch]['Recall@50'] > self.best_recall50:
            print("New best for Recall@50")
            self.model.save_weights(self.rsmodel.name + "_best_recall_50/" + self.rsmodel.name)
            self.best_recall50 = self.eval_metrics[self.epoch]['Recall@50']
            self.best_recall50_epoch = self.epoch
            with open(self.rsmodel.name + "_best_recall_50/" + "epoch.txt", "w") as text_file:
                text_file.write(str(self.best_recall50_epoch))

    def ncdg_100_watch(self):
        if self.eval_metrics[self.epoch]['NCDG@100'] > self.best_ncdg100:
            print("New best for NCDG@100")
            self.model.save_weights(self.rsmodel.name + "_best_ncdg_100/" + self.rsmodel.name)
            self.best_ncdg100 = self.eval_metrics[self.epoch]['NCDG@100']
            self.best_ncdg100_epoch = self.epoch
            with open(self.rsmodel.name + "_best_ncdg_100/" + "epoch.txt", "w") as text_file:
                text_file.write(str(self.best_ncdg100_epoch))

    def on_train_end(self, logs=None):
        self.plot_history()

    def get_history_df(self):

        outt1 = {
            'epochs': [x for x in self.rsmodel.mc.loss_metrics.keys()]
        }

        outt2 = {
            'epochs': [x for x in self.rsmodel.mc.eval_metrics.keys()]
        }

        for k in self.loss_metrics[1].keys():
            outt1[k] = [self.loss_metrics[x][k] for x in self.loss_metrics.keys()]

        for k in self.eval_metrics[1].keys():
            outt2[k] = [self.eval_metrics[x][k] for x in self.eval_metrics.keys()]

        self.history_loss_df = pd.DataFrame(outt1)
        self.history_loss_df.to_json(self.rsmodel.name + "_loss.json")

        self.history_df = pd.DataFrame(outt2)
        self.history_df.to_json(self.rsmodel.name + "_metrics.json")

        return self.history_df

    def plot_history(self):
        return self.get_history_df().set_index(self.history_df.epochs, drop=True).iloc[:, 1:].plot(figsize=(20, 10))

    def calc_tsne(self):
        num_words = self.rsmodel.dataset.num_words
        input_single_item_matrix = np.zeros((num_words, num_words))
        np.fill_diagonal(input_single_item_matrix, 1.)
        qqq = scale_d(self.model.predict(input_single_item_matrix)).numpy() * .99
        np.fill_diagonal(qqq, 1.)
        tsne_coordinates = TSNE(n_components=2, metric="precomputed", angle=0.5, perplexity=30, random_state=6).fit(
            (1 - qqq))
        tsne_coordinates = tsne_coordinates.embedding_
        self.tsne_df.loc[self.epoch] = [self.epoch, tsne_coordinates]
        self.tsne_df.to_json(self.rsmodel.name + "_tsne.json")
