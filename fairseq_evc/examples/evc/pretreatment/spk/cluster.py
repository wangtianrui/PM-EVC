import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from tqdm import tqdm
import os
import pickle
import json
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import argparse
from collections import OrderedDict
import concurrent.futures as cf
import kaldiio
import torch
import numpy as np
import scipy.linalg
from sklearn.cluster._kmeans import k_means
from wespeaker.utils.utils import validate_path
from copy import copy

def batch_matrix_multiply(A, B, batch_size):
    n, m = A.shape[0], B.shape[1]
    C = np.zeros((n, m))
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            C[i:end_i, j:end_j] = np.dot(A[i:end_i, :], B[:, j:end_j])
    return C

def cluster(embeddings, p=.01, num_spks=None, min_num_spks=1, max_num_spks=20):
    # Define utility functions
    def cosine_similarity(M):
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        temp = batch_matrix_multiply(M, M.T, 1000)
        return temp

    def prune(M, p):
        m = M.shape[0]
        if m < 1000:
            n = max(m - 10, 2)
        else:
            n = int((1.0 - p) * m)

        for i in range(m):
            indexes = np.argsort(M[i, :])
            low_indexes, high_indexes = indexes[0:n], indexes[n:m]
            M[i, low_indexes] = 0.0
            M[i, high_indexes] = 1.0
        return 0.5 * (M + M.T)

    def laplacian(M):
        M[np.diag_indices(M.shape[0])] = 0.0
        D = np.diag(np.sum(np.abs(M), axis=1))
        return D - M

    def spectral(M, num_spks, min_num_spks, max_num_spks):
        eig_values, eig_vectors = scipy.linalg.eigh(M)
        num_spks = num_spks if num_spks is not None \
            else np.argmax(np.diff(eig_values[:max_num_spks + 1])) + 1
        num_spks = max(num_spks, min_num_spks)
        return eig_vectors[:, :num_spks]

    def kmeans(data):
        k = data.shape[1]
        # centroids, labels = scipy.cluster.vq.kmeans2(data, k, minit='++')
        _, labels, _ = k_means(data, k, random_state=None, n_init=10)
        return labels

    # Fallback for trivial cases
    if len(embeddings) <= 2:
        return [0] * len(embeddings)

    # Compute similarity matrix
    with torch.no_grad():
        similarity_matrix = cosine_similarity(embeddings) # B, D
        # Prune matrix with p interval
        print("start 1")
        pruned_similarity_matrix = prune(similarity_matrix, p) # B, B
        # Compute Laplacian
        print("start 2")
        laplacian_matrix = laplacian(pruned_similarity_matrix) # B, B
        # Compute spectral embeddings
        print("start 3")
        spectral_embeddings = spectral(laplacian_matrix, num_spks, min_num_spks, max_num_spks) # B, spk
        # Assign class labels
        print("start 4")
        labels = kmeans(spectral_embeddings)
        return labels

if __name__ == "__main__":
    infos = {}
    spk_embs = {}
    all_spks = []
    with open(r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/all_info.tsv", "r") as rf:
        with open(r"/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb/km_models/all_info.tsv", "w") as wf:
            lines = [i.strip() for i in rf.readlines()]
            for line in tqdm(lines):
                temp = line.split("\t")
                if len(temp) != 8:
                    (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
                else:
                    path, sr, length, spk, emo, level, o_trans, emo2 = temp
                npy_path = path.replace(
                    "/CDShare2/2023/wangtianrui/dataset/emo",
                    "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb"
                ).replace(
                    "/CDShare3/2023/wangtianrui",
                    "/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb"
                ).split(".")[0] + ".npy"
                all_spks.append(spk)
                if spk.split("|")[1] != "_" and spk.split("|")[1] != "Unknown":
                    print(line, file=wf)
                    continue
                data_name = spk.split("|")[0]
                if data_name not in spk_embs:
                    spk_embs[data_name] = [np.load(npy_path)]
                    infos[data_name] = [line]
                else:
                    spk_embs[data_name].append(np.load(npy_path))
                    infos[data_name].append(line)
    
    print(spk_embs.keys())
    rest_infos = []
    rest_spk_emb = []
    data_names = []
    for data_name in spk_embs.keys(): # only iemocap miss spk
        rest_infos = rest_infos + infos[data_name]
        rest_spk_emb = rest_spk_emb + spk_embs[data_name]
        data_names = data_names + [data_name, ] * len(spk_embs[data_name])
        
    rest_spk_emb = np.stack(rest_spk_emb, axis=0)
    print(rest_spk_emb.shape)
    
    spk_num = 100
    temp_all_spks = copy(all_spks)
    labels = cluster(rest_spk_emb,
                    num_spks=spk_num,
                    min_num_spks=5,
                    max_num_spks=100)
    # print(labels)
    label_infos = {}
    with open("/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb/km_models/missed_%d.tsv"%(spk_num), "w") as wf:
        with open("/CDShare2/2023/wangtianrui/dataset/emo/english_emo_data/spk_emb/km_models/spkdict_%d.json"%(spk_num), "w") as wf2:
            for i in range(rest_spk_emb.shape[0]):
                temp = rest_infos[i].split("\t")
                if len(temp) != 8:
                    (path, sr, length, spk, emo, level), o_trans, emo2 = temp[:6], " ".join(temp[7:-1]), temp[-1]
                else:
                    path, sr, length, spk, emo, level, o_trans, emo2 = temp
                spk = data_names[i]+"_clustr|"+str(labels[i])
                temp_all_spks.append(spk)
                print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"%(path, sr, length, spk, emo, level, o_trans, emo2), file=wf)
            unique_spks = list(set(temp_all_spks))
            spk_dict = {spk: idx for idx, spk in enumerate(unique_spks)}
            json.dump(spk_dict, wf2)
        