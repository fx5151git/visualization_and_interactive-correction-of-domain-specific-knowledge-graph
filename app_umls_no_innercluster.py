from flask import Flask, render_template, url_for, request,session,redirect
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import json
import scipy
from numpy import dot, array, zeros, setdiff1d
from numpy.linalg import norm
from numpy.random import shuffle
from scipy.io.matlab import loadmat
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.cluster import KMeans
from rescal import rescal_als
# import matplotlib.pyplot as plt
from bisect import bisect
from flask_session import Session
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE
import logging

app = Flask(__name__)
app.secret_key = 'why would I tell you my secret key?'

app.config['SESSION_TYPE'] = 'filesystem'
TEMPLATES_AUTO_RELOAD = True

sess = Session()
sess.init_app(app)

handler = logging.FileHandler('flask.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]- %(message)s")
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

def predict_rescal_als(T):
    A, R, f, itr, exectimes = rescal_als(
        T, 10, init='nvecs',
        lambda_A= 10, lambda_R= 10
    )
    n = A.shape[0]
    P = zeros((n, n, len(R)))
    for k in range(len(R)):
        P[:, :, k] = dot(A, dot(R[k], A.T))
    return A,P,R


def innerfold(T, mask_idx, target_idx, e, k, sz, GROUND_TRUTH):
    Tc = [Ti.copy() for Ti in T]
    # mask_idx = np.unravel_index(mask_idx, (e, e, k))
    # target_idx = np.unravel_index(target_idx, (e, e, k))
    mask_idx = np.unravel_index(mask_idx, (k, e, e))
    target_idx = np.unravel_index(target_idx, (e, e, k))

    for i in range(len(mask_idx[0])):
        Tc[mask_idx[0][i]][mask_idx[1][i], mask_idx[2][i]] = 0
    # set values to be predicted to zero
    # for i in mask_idx:
    #     for m in range(e):
    #         for n in range(e):
    #             Tc[m][n][i] = 0

    # predict unknown values
    entity_embedding, P ,R = predict_rescal_als(Tc)
    # P = normalize_predictions(P, e, k)

    # compute area under precision recall curve
    prec, recall, _ = precision_recall_curve(GROUND_TRUTH[target_idx], P[target_idx])
    # print("test",auc(recall, prec))
    return entity_embedding, P, R, auc(recall, prec), T


def get_outlier_scores(tensor, GROUND_TRUTH, e, k, mask_idx):
    test = GROUND_TRUTH.copy()
    mask_idx = np.unravel_index(mask_idx, (k, e, e))
    for i in range(len(mask_idx[0])):
        test[mask_idx[1][i]][mask_idx[2][i]][mask_idx[0][i]] = 0

    rel_index_1 = []
    distribution_1_all = []
    distribution_1_visualize = []
    outlier_num = []
    distribution_0_all = []

    for t in range(k):
        distribution_1 = []
        distribution= []
        for m in range(e):
            for n in range(e):
                distribution.append(tensor[m,n,t])
                if (test[m, n, t] == 1):
                    distribution_1.append(tensor[m, n, t])
        if (len(distribution_1) > 0):
            temp = np.percentile(distribution_1, 25)
            distribution_1_all.append([temp, len(distribution_1)])
            distribution.sort()
            index_0 = bisect(distribution, temp)
            distribution_0_all.append(index_0 / len(distribution))
        else:
            distribution_1_all.append([0, 1])
            distribution.sort()
            index_0 = bisect(distribution, 0)
            distribution_0_all.append(index_0 / len(distribution))


    for t in range(k):
        number = 0
        for m in range(e):
            for n in range(e):
                if (test[m, n, t] == 0 and tensor[m, n, t] > distribution_1_all[t][0]):
                    number += 1
        outlier_num.append(number / distribution_1_all[t][1])

    dist_mean = np.mean(outlier_num)
    dist_std = np.std(outlier_num)
    # max_outlier  = np.percentile(outlier_num,95)
    # chose = []
    # for i in outlier_num:
    #     if (i > 0  and i < max_outlier):
    #         chose.append(i)

    outlier_num_re = [(i - dist_mean) / np.std(outlier_num) for i in outlier_num]

    fx = []
    for i in np.unique(mask_idx[0]):
        fx.append(outlier_num_re[i])

    max = np.percentile(outlier_num_re, 95)
    chose = []
    for i in outlier_num_re:
        if (i > (0 - dist_mean) / np.std(outlier_num) and i < max):
            chose.append(i)

    chose_1 = [np.abs(i - np.mean(chose)) for i in chose]
    re_chose = sorted(chose_1)


    for i in range(len(chose)):
        rel_index_1.append(chose_1.index(re_chose[i]))

    return rel_index_1, re_chose, distribution_0_all

def initiate_data():

    mat = loadmat('D:/knowledge graph/rescal.py-master/data/uml.mat')
    entity_datafile = 'D:/knowledge graph/rescal.py-master/data/uml_entities.txt'
    relation_datafile = 'D:/knowledge graph/rescal.py-master/data/uml_relations.txt'

    entities = []
    relations = []

    with open(entity_datafile, 'r') as f:
        for line in f.readlines():
            num, entity = line.split('\t')
            entities.append(entity.strip())

    with open(relation_datafile, 'r') as f:
        for line in f.readlines():
            num, relation = line.split('\t')
            relations.append(relation.strip())

    K = array(mat['Rs'], np.float32)
    K[np.isnan(K)] = 0
    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k

    # copy ground truth before preprocessing
    GROUND_TRUTH = K.copy()

    # construct array for rescal
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    # Do relation cross-validation
    Percentage = 5
    ID_relation = list(range(k))
    # ID_relation.pop(5)
    shuffle(ID_relation)
    IDX = list(range(SZ))
    rel_percentage = int(e * e / Percentage)
    #
    frelation = 4
    offset = 0
    relation_val = []

    idx_test = []
    idx_relation = ID_relation[:frelation]
    #
    # relation_val.append(ID_relation[offset:offset + frelation])
    for rel in idx_relation:
        print(relations[rel])
        random_list = IDX[rel * e * e:(rel + 1) * e * e]
        shuffle(random_list)
        idx_test = idx_test + random_list[:rel_percentage]
    # FOLDS = 10
    # IDX = list(range(SZ))
    # shuffle(IDX)
    #
    # fsz = int(SZ / FOLDS)
    # offset = 0
    # idx_test = IDX[offset:offset + fsz]

    # np.save("idx_test_umls_rel.npy",idx_test)
    # np.save("idx_umls_rel.npy", IDX)
    idx_test = np.load("idx_test_umls_rel.npy")
    IDX = np.load("idx_umls_rel.npy")

    entity_embedding, tensor, rel_embedding, auc, Tc = innerfold(T, idx_test, IDX, e, k, SZ, GROUND_TRUTH)
    bar_index, bar_value, distribution_0_visualize = get_outlier_scores(tensor, GROUND_TRUTH, e, k, idx_test)

    tensor_vis =  tensor.copy()

    for m in range(tensor.shape[2]):
        for n in range(tensor.shape[0]):
            for t in range(tensor.shape[1]):
                if(tensor[n,t,m]<0):
                    tensor_vis[n,t,m] = 0
                if(Tc[m][n,t] == 1):
                    tensor_vis[n,t,m] = tensor[n,t,m] + 1


    cluster = len(entities)
    re_tensor, entities, last, re_Ground_truth, re_tensor_ori = entity_cluster_tensor(cluster,entity_embedding, tensor_vis, entities, GROUND_TRUTH, tensor)

    return re_tensor, bar_index, bar_value, entities, relations, GROUND_TRUTH, distribution_0_visualize, last, re_Ground_truth, entity_embedding

def generate_data(cluster):

    mat = loadmat('D:/knowledge graph/rescal.py-master/data/uml.mat')
    entity_datafile = 'D:/knowledge graph/rescal.py-master/data/uml_entities.txt'
    relation_datafile = 'D:/knowledge graph/rescal.py-master/data/uml_relations.txt'

    entities = []
    relations = []

    with open(entity_datafile, 'r') as f:
        for line in f.readlines():
            num, entity = line.split('\t')
            entities.append(entity.strip())

    with open(relation_datafile, 'r') as f:
        for line in f.readlines():
            num, relation = line.split('\t')
            relations.append(relation.strip())

    K = array(mat['Rs'], np.float32)
    K[np.isnan(K)] = 0
    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k

    # copy ground truth before preprocessing
    GROUND_TRUTH = K.copy()

    # construct array for rescal
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    # Do relation cross-validation
    # Percentage = 5
    # ID_relation = list(range(k))
    # ID_relation.pop(5)
    # shuffle(ID_relation)
    # IDX = list(range(SZ))
    # rel_percentage = int(e * e / Percentage)
    #
    # frelation = 4
    # offset = 0
    # relation_val = []
    #
    # idx_test = []
    # idx_relation = ID_relation[:frelation]
    #
    # relation_val.append(ID_relation[offset:offset + frelation])
    # for rel in idx_relation:
    #     print(relations[rel])
    #     random_list = IDX[rel * e * e:(rel + 1) * e * e]
    #     shuffle(random_list)
    #     idx_test = idx_test + random_list[:rel_percentage]
    FOLDS = 10
    # IDX = list(range(SZ))
    # shuffle(IDX)
    #C
    # fsz = int(SZ / FOLDS)
    # offset = 0
    idx_test = np.load("idx_test_umls_rel.npy")
    IDX = np.load("idx_umls_rel.npy")

    entity_embedding, tensor, rel_embedding, auc,Tc = innerfold(T, idx_test, IDX, e, k, SZ, GROUND_TRUTH)
    bar_index, bar_value, distribution_0_visualize = get_outlier_scores(tensor, GROUND_TRUTH, e, k, idx_test)

    tensor_vis =  tensor.copy()

    for m in range(tensor.shape[2]):
        for n in range(tensor.shape[0]):
            for t in range(tensor.shape[1]):
                if(tensor[n,t,m]<0):
                    tensor_vis[n,t,m] = 0
                if(Tc[m][n,t] == 1):
                    tensor_vis[n,t,m] = tensor[n,t,m] + 1

    # 12
    re_tensor, entities, last, re_Ground_truth, re_tensor_ori = entity_cluster_tensor(cluster,entity_embedding, tensor_vis, entities, GROUND_TRUTH, tensor)

    return re_tensor, bar_index, bar_value, entities, relations, GROUND_TRUTH, distribution_0_visualize, last, re_Ground_truth, entity_embedding, re_tensor_ori


def entity_cluster_tensor(clusters, embedding, tensor,entities, ground_truth, tensor_ori):
    # kmeans for error cluster
    # color = np.random.rand(clusters)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(embedding)

    labels = kmeans.labels_

    index = []
    last = []
    temp = 0
    for i in range(clusters):
        indices = [m for m, x in enumerate(labels) if x == i]
        temp = temp + len(indices)
        last.append(temp-1)
        index = index + indices

    entities = [entities[i] for i in index]
    result = tensor.copy()
    result_1 = tensor_ori.copy()
    re_ground_truth = ground_truth.copy()
    for i in range(tensor.shape[2]):
        result[:, :, i] = tensor[index, :, i]
        result[:, :, i] = result[:, index, i]
        re_ground_truth[:, :, i] = ground_truth[index, :, i]
        re_ground_truth[:, :, i] = re_ground_truth[:, index, i]
        result_1[:, :, i] = tensor_ori[index, :, i]
        result_1[:, :, i] = result_1[:, index, i]

    return result, entities, last, re_ground_truth, result_1


def create_plot( P , entities, relation, distribution_0, entityX, entityY, last, modifyX, modifyY, questionX, questionY):
    r_start = 0
    c_start = 0
    entities_x = []
    entities_y = []
    # #
    heat_data = P[:,:,relation]
    test = heat_data.copy()

    for r in last:
        temp_1 = heat_data[r_start:r + 1,:]
        print(r_start, r)
        dendro = ff.create_dendrogram(temp_1, orientation='bottom')

        dendro_leaves = dendro['layout']['xaxis']['ticktext']
        dendro_leaves = list(map(int, dendro_leaves))

        # heat_data[r_start:r+1,:] = heat_data[r_start:r+1,:][dendro_leaves, :]
        for i,j in zip(dendro_leaves,range(r_start,r+1)):
            test[j,:] = temp_1[i,:]
        temp = entities[r_start:r + 1]
        entities_x = entities_x + [temp[i] for i in dendro_leaves]

        r_start = r+1


    heat_data_test = np.swapaxes(test,0,1)
    for c in last:
        temp_2 = heat_data_test[c_start: c + 1,: ]
        dendro = ff.create_dendrogram(temp_2, orientation='bottom')

        dendro_leaves = dendro['layout']['xaxis']['ticktext']
        dendro_leaves = list(map(int, dendro_leaves))


        for i,j in zip(dendro_leaves,range(r_start,r+1)):
            test[:,j] = temp_2[i,:]

        temp = entities[c_start: c + 1]
        entities_y = entities_y+ [temp[i] for i in dendro_leaves]

        c_start = c + 1

    # dendro = ff.create_dendrogram(P[:, :, relation], orientation='bottom')
    #
    # dendro_leaves = dendro['layout']['xaxis']['ticktext']
    # dendro_leaves = list(map(int, dendro_leaves))
    #
    # heat_data = P[:, :, relation][dendro_leaves, :]
    # heat_data = heat_data[:, dendro_leaves]
    #
    # entities = [entities[i] for i in dendro_leaves]
    # print(last)
    final = []
    #
    for i in range(len(entities)):
        new_row = []
        for j in range(len(entities)):
            new_row.append(test[j, i])
        final.append(list(new_row))

    data = [go.Heatmap(
            x=entities_x,
            y=entities,
            z=final,
            xgap=1,
            ygap=1,
            colorscale=[
                [0, "rgb(222,239,242)"],
                [0.49999999, "rgb(26,63,246)"],
                [0.5,"rgb(246,77,26)"],
                [1,  "rgb(242,240,247)"],

        ],
        colorbar= dict(x= -.1, len= 0.5))]


    fig = go.Figure(data=data)
    #

    # 230
    fig['layout'].update({
        # 'autosize': True,
        'width': 900, 'height': 600,
                             'margin':dict(
                                        l=10,
                                        r=100,
                                        b=10,
                                        t=150,
                                        pad=4
                                    ),
                             'showlegend': False,
                             'hovermode': 'closest',
                             'xaxis' : dict(tickfont= dict(
                                                size=8,
                                                color='black'
                                            ),
                                            tickangle = 40,
                                            side = 'top'),
                             'yaxis': dict(tickfont=dict(
                                                size=8,
                                                color='black'
                                            ),
                                            side = 'right'),

                            # 'shapes' :[
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[0],
                            #         x1=entities[-1],
                            #         y1=last[0],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[1],
                            #         x1=entities[-1],
                            #         y1=last[1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[2],
                            #         x1=entities[-1],
                            #         y1=last[2],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[3],
                            #         x1=entities[-1],
                            #         y1=last[3],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[4],
                            #         x1=entities[-1],
                            #         y1=last[4],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[5],
                            #         x1=entities[-1],
                            #         y1=last[5],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[6],
                            #         x1=entities[-1],
                            #         y1=last[6],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[7],
                            #         x1=entities[-1],
                            #         y1=last[7],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[8],
                            #         x1=entities[-1],
                            #         y1=last[8],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[9],
                            #         x1=entities[-1],
                            #         y1=last[9],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=entities[0],
                            #         y0=last[10],
                            #         x1=entities[-1],
                            #         y1=last[10],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[0],
                            #         y0=entities[0],
                            #         x1=last[0],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[1],
                            #         y0=entities[0],
                            #         x1=last[1],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[2],
                            #         y0=entities[0],
                            #         x1=last[2],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[3],
                            #         y0=entities[0],
                            #         x1=last[3],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[4],
                            #         y0=entities[0],
                            #         x1=last[4],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[5],
                            #         y0=entities[0],
                            #         x1=last[5],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[6],
                            #         y0=entities[0],
                            #         x1=last[6],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[7],
                            #         y0=entities[0],
                            #         x1=last[7],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[8],
                            #         y0=entities[0],
                            #         x1=last[8],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[9],
                            #         y0=entities[0],
                            #         x1=last[9],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     ),
                            #     go.layout.Shape(
                            #         type="line",
                            #         x0=last[10],
                            #         y0=entities[0],
                            #         x1=last[10],
                            #         y1=entities[-1],
                            #         opacity=0.3,
                            #         line=dict(
                            #             color="Crimson",
                            #             width=1,
                            #         )
                            #     )
                            # ]
                             })
    shapes = []
    for i in last:
        shapes.append(
                go.layout.Shape(
                    type="line",
                    x0=-0.5,
                    y0=i + 0.5,
                    x1=len(entities)-0.5,
                    y1=i + 0.5,
                    opacity=0.5,
                    line=dict(
                        color="Crimson",
                        width=2,
                    )
                )
            )
        shapes.append(go.layout.Shape(
                    type="line",
                    x0=i + 0.5,
                    y0= -0.5,
                    x1= i + 0.5,
                    y1= len(entities)-0.5,
                    opacity=0.5,
                    line=dict(
                        color="Crimson",
                        width=2,
                    )
                ))

    fig['layout'].update({
        'shapes': shapes
        })

    image_modify = []
    if modifyY != None:
        for i, j in zip(modifyX,modifyY):
            image_modify.append(go.layout.Image(
                        source="static/pen.png",
                        xref="x",
                        yref="y",
                        x=i,
                        y=j,
                        sizex=1,
                        sizey=1,
                        xanchor="center",
                        yanchor="middle",
                        sizing="stretch",
                        opacity=1,
                        layer="above"))

    if questionX != None :
        for i, j in zip(questionX, questionY):
            image_modify.append(go.layout.Image(
                            source="static/quesionmark.png",
                            xref="x",
                            yref="y",
                            x=i,
                            y=j,
                            sizex=1,
                            sizey=1,
                            sizing="stretch",
                            xanchor="center",
                            yanchor="middle",
                            opacity=1,
                            layer="above"))


    fig['layout'].update({
        'images': image_modify
    })


    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return fig

def create_bar_plot(relations,bar_index, bar_value):
    relation_chosen = []
    for i in bar_index:
        relation_chosen.append(relations[i])

    data = [go.Bar(
        x=bar_value,
        y=relation_chosen,
        orientation='h'
    )]
    layout = go.Layout(
                          width= 350,
                          # height= 600,
                          margin = dict(
                                        l=150,
                                        b=20,
                                        t=10,
                                        # pad=4
                                    ))
    fig = go.Figure(data=data, layout = layout)

    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return fig, relation_chosen

def create_bar_plot_highlight(relations,bar_index, bar_value, relation):
    relation_chosen = []
    for i in bar_index:
        relation_chosen.append(relations[i])

    # print(relation)
    #
    # # colors = ['lightslategray', ] * len(relation_chosen)
    # # colors[relation] = 'crimson'
    #
    # print(colors)

    data = [go.Bar(
        x=bar_value,
        y=relation_chosen,
        orientation='h'
    )]
    layout = go.Layout(
                          width= 350,
                          # height= 600,
                          margin = dict(
                                        l=150,
                                        b=20,
                                        t=10,
                                        # pad=4
                                    ))
    fig = go.Figure(data=data, layout = layout)

    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return fig, relation_chosen

def get_true_entity(original, relation, entities):
    entityX = []
    entityY = []

    for n in range(original.shape[0]):
        for t in range(original.shape[1]):
            if(original[n,t,relation] == 1):
                entityX.append(entities[n])
                entityY.append(entities[t])

    return entityX,entityY

def create_line_plot(entities):
    Sum_of_squared_distances = []
    K = range(1, int(len(entities)/4))
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(entities)
        Sum_of_squared_distances.append(km.inertia_)
    data = [
        go.Scatter(
                x=np.array(K),
                y=Sum_of_squared_distances,
                mode="lines+markers",
                marker=dict(size=4.5,
                                color=0),
                showlegend=False)
        ]

    layout = go.Layout(
                          width= 600,
                          height= 400
                          # margin = dict(
                          #               l=150,
                          #               b=20,
                          #               t=10,
                          #               # pad=4
                          #           )
    )

    fig = go.Figure(data=data, layout = layout)

    return fig

def create_tsne_plot(entity_embedding,entities,cluster, perp, set):

    if cluster != 0 and set == True:
        entity_tsne = session.get('entity_tsne', None)
        # tsne = TSNE(perplexity=perp, n_components=2, init='pca', n_iter=3000)
        # entity_tsne = tsne.fit_transform(entity_embedding)
        color = np.random.rand(cluster)

        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(entity_embedding)

        labels = kmeans.labels_
        data = [
            go.Scatter(
                x=entity_tsne[:, 0],
                y=entity_tsne[:, 1],
                mode='markers',
                text=entities,
                textposition='top center',
                marker=dict(color=labels)
                            )
        ]
    else:
        tsne = TSNE(perplexity=perp, n_components=2, init='pca', n_iter=3000)
        entity_tsne = tsne.fit_transform(entity_embedding)
        session['entity_tsne'] = entity_tsne
        data = [
            go.Scatter(
                x=entity_tsne[:, 0],
                y=entity_tsne[:, 1],
                mode='markers',
                text=entities,
                textposition='top center'
            )
        ]

    layout = go.Layout(
                          width= 600,
                          height= 500
                          # margin = dict(
                          #               l=150,
                          #               b=20,
                          #               t=10,
                          #               # pad=4
                          #           )
    )

    fig = go.Figure(data=data, layout = layout)

    return fig







def get_accuracy(ori_matrix, re_matrix,id,e,k):
    id = np.unravel_index(id, (e, e, k))
    prec, recall, _ = precision_recall_curve(ori_matrix[id], re_matrix[id])


    return roc_auc_score(ori_matrix[id], re_matrix[id]), auc(recall, prec)



@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    matrix, bar_index, bar_value, entities, relations, GROUND_TRUTH, distribution_1, last, re_ground_truth, entity_embedding = initiate_data()
    session['matrix'] = matrix.tolist()
    session['bar_index'] = bar_index
    session['bar_value'] = bar_value
    session['relations'] = relations
    session['Ground_truth'] = GROUND_TRUTH.tolist()
    session['distribution_1'] =  distribution_1
    session['last'] = last
    session['re_ground_truth'] = re_ground_truth.tolist()
    session['entities'] = entities
    session['entity_embedding'] = entity_embedding
    session['perp'] = 5
    session['num'] = 0
    lineplot = create_line_plot(entity_embedding)
    tsne_plot = create_tsne_plot(entity_embedding, entities,0, session['perp'], True)
    graphJSON = [lineplot, tsne_plot]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('cluster.html', graphJSON=graphJSON)

@app.route('/perplexity', methods=['POST'])
def perplexity():
    entities = session.get('entities', None)
    entity_embedding = session.get('entity_embedding', None)
    num = session.get('num', None)

    perp = int(request.form['perp'])
    session['perp'] = perp
    tsne_plot = create_tsne_plot(entity_embedding, entities, num, perp, False )
    graphJSON = [tsne_plot]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/refresh', methods=['POST'])
def refresh():
    entities = session.get('entities', None)
    entity_embedding = session.get('entity_embedding', None)
    num = session.get('num', None)
    perp = session.get('perp', None)

    tsne_plot = create_tsne_plot(entity_embedding, entities, num, perp, False)
    graphJSON = [tsne_plot]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/predict', methods=['POST'])
def predict():
    print('test')
    cluster= int(request.form['submit'])
    matrix = np.array(session.get('matrix', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)
    relations = session.get('relations',None)
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    distribution_1 = session.get('distribution_1',None)
    last = session.get('last',None)
    re_ground_truth = np.array(session.get('re_ground_truth',None))
    operation =  {k: {"mX":[],"mY": [], "qX":[], "qY":[]} for k in relations}

    bar, relation_chosen = create_bar_plot(relations, bar_index, bar_value)
    relation = relations.index(relation_chosen[-1])
    session['relation'] = relation
    session['relation_chosen'] = relation_chosen
    relation_left = [i for i in relations if i not in relation_chosen]
    entityX, entityY = get_true_entity(re_ground_truth, relation, entities)
    matrix, bar_index, bar_value, entities, relations, GROUND_TRUTH, distribution_1, last, re_ground_truth, entity_embedding, matrix_ori = generate_data(cluster)
    session['matrix'] = matrix.tolist()
    session['matrix_ori'] = matrix_ori.tolist()
    session['bar_index'] = bar_index
    session['bar_value'] = bar_value
    session['relations'] = relations
    session['Ground_truth'] = GROUND_TRUTH.tolist()
    session['distribution_1'] = distribution_1
    session['last'] = last
    session['re_ground_truth'] = re_ground_truth.tolist()
    session['entities'] = entities
    session['entity_embedding'] = entity_embedding
    session['operation'] = operation

    IDX = np.load("idx_umls_rel.npy")
    AUC_Roc, auc_PR = get_accuracy(re_ground_truth, matrix_ori,IDX, GROUND_TRUTH.shape[0], GROUND_TRUTH.shape[2])
    app.logger.debug("AUC_ROC: %f ; AUC_PR: %f " %(AUC_Roc, auc_PR))


    heatmap = create_plot( matrix, entities, relation, distribution_1[relation], entityX, entityY, last,operation[relation_chosen[-1]]["mX"], operation[relation_chosen[-1]]["mY"], operation[relation_chosen[-1]]["qX"], operation[relation_chosen[-1]]["qY"])
    graphJSON = [heatmap, bar]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)

    session['entityX'] = entityX
    session['entityY'] = entityY

    session['cluster'] = cluster

    return render_template('result_2.html', graphJSON=graphJSON, relations_left=relation_left,relations = relations, relation=relation, relation_chosen=relation_chosen)


@app.route('/configuration', methods=['POST'])
def configuration():
    relation = request.form['relation']
    matrix = np.array(session.get('matrix', None))

    matrix_ori = np.array(session.get('matrix_ori', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)
    relations = session.get('relations',None)
    relation_chosen = session.get('relation_chosen',None)
    relation_left = [i for i in relations if i not in relation_chosen]
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    distribution_1 = session.get('distribution_1',None)
    last = session.get('last',None)
    re_ground_truth = np.array(session.get('re_ground_truth',None))
    operation = session.get('operation', None)
    modifyX = operation[relation]["mX"]
    modifyY = operation[relation]["mY"]
    questionX =  operation[relation]["qX"]
    questionY =  operation[relation]["qY"]

    relation = relations.index(relation)
    session['relation'] = relation

    entityX, entityY = get_true_entity(re_ground_truth, session['relation'],entities)

    session['entityX'] = entityX
    session['entityY'] = entityY

    # modifyX = session.get('modifyX', None)
    # modifyY = session.get('modifyY', None)
    # questionX = session.get('questionX', None)
    # questionY = session.get('questionY', None)



    heatmap = create_plot( matrix, entities, relation, distribution_1[relation],entityX, entityY, last, modifyX, modifyY, questionX, questionY)
    bar, relation_chosen = create_bar_plot(relations, bar_index, bar_value)
    graphJSON = [heatmap, bar]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('result_2.html', graphJSON=graphJSON, relations_left=relation_left, relations = relations, relation = relation,relation_chosen=relation_chosen)


# @app.route('/configuration', methods=['POST'])
# def configuration():
#     relation = request.form['relation']
#     matrix = np.array(session.get('matrix', None))
#     bar_index = session.get('bar_index', None)
#     bar_value = session.get('bar_value' , None)
#     entities = session.get('entities', None)
#     relations = session.get('relations',None)
#     relation_chosen = session.get('relation_chosen', None)
#     GROUND_TRUTH = np.array(session.get('Ground_truth',None))
#     distribution_1 = session.get('distribution_1',None)
#     last = session.get('last', None)
#     re_ground_truth = np.array(session.get('re_ground_truth',None))
#     operation = session.get('operation', None)
#
#     modifyX = operation[relation]["mX"]
#     modifyY = operation[relation]["mY"]
#     questionX =  operation[relation]["qX"]
#     questionY =  operation[relation]["qY"]
#
#     relation = relations.index(relation)
#     session['relation'] = relation
#
#     entityX, entityY = get_true_entity(re_ground_truth, session['relation'], entities)
#
#     session['entityX'] = entityX
#     session['entityY'] = entityY
#
#     heatmap = create_plot(GROUND_TRUTH, matrix , entities, relation, distribution_1[relation], entityX, entityY, last,modifyX, modifyY, questionX, questionY)
#     bar = create_bar_plot(relations,bar_index, bar_value)
#     graphJSON = [heatmap, bar]
#     data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
#     return data
#
#

@app.route('/frombar', methods=['POST'])
def frombar():
    relation =int(request.form['relation'])
    matrix = np.array(session.get('matrix', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)
    relations = session.get('relations',None)
    relation_chosen = session.get('relation_chosen', None)
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    distribution_1 = session.get('distribution_1',None)
    last = session.get('last', None)
    re_ground_truth = np.array(session.get('re_ground_truth',None))
    operation = session.get('operation', None)
    matrix_ori = np.array(session.get('matrix_ori', None))
    bar = create_bar_plot_highlight(relations, bar_index, bar_value, relation)

    temp = relation_chosen[relation]
    relation = relations.index(temp)
    session['relation'] = relation

    entityX, entityY = get_true_entity(re_ground_truth, session['relation'], entities)

    session['entityX'] = entityX
    session['entityY'] = entityY

    modifyX = operation[temp]["mX"]
    modifyY = operation[temp]["mY"]
    questionX = operation[temp]["qX"]
    questionY = operation[temp]["qY"]

    IDX = np.load("idx_umls_rel.npy")
    AUC_Roc, auc_PR = get_accuracy(re_ground_truth, matrix_ori,IDX,GROUND_TRUTH.shape[0], GROUND_TRUTH.shape[2])
    app.logger.debug("AUC_ROC: %f ; AUC_PR: %f " %(AUC_Roc, auc_PR))

    heatmap= create_plot( matrix , entities, relation, distribution_1[relation], entityX, entityY, last,modifyX, modifyY, questionX, questionY)
    # bar = create_bar_plot(relations,bar_index, bar_value)
    # session["entities_x"] = entities_x
    # session['entities_y'] = entities_y
    graphJSON = [heatmap, bar]
    data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return data

@app.route('/change_to_true', methods=['POST'])
def change_to_true():
    x = request.form['x']
    y = request.form['y']
    x = eval(x);
    y = eval(y);
    relation = session['relation']
    matrix = np.array(session.get('matrix', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)

    relations = session.get('relations',None)
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    last = session.get('last', None)
    distribution_1 = session.get('distribution_1',None)
    operation = session.get('operation', None)
    matrix_ori = np.array(session.get('matrix_ori', None))

    entityX = session.get('entityX',None)
    entityY = session.get('entityY', None)
    entityX.append(x)
    entityY.append(y)
    session['entityX'] = entityX
    session['entityY'] = entityY

    x = entities.index(x)
    y = entities.index(y)


    matrix[x,y,relation] = 2
    matrix_ori[x,y,relation] = 1


    session['matrix'] = matrix.tolist()
    session['matrix_ori'] = matrix_ori.tolist()
    re_ground_truth = np.array(session.get('re_ground_truth', None))
    print(re_ground_truth[x,y,relation])


    operation[relations[relation]]["mX"].append(x)
    operation[relations[relation]]["mY"].append(y)

    heatmap= create_plot( matrix , entities, relation, distribution_1[relation], entityX, entityY,last,operation[relations[relation]]["mX"],operation[relations[relation]]["mY"],operation[relations[relation]]["qX"],operation[relations[relation]]["qY"] )

    session['operation'] = operation
    graphJSON = [heatmap]
    data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)

    return data


@app.route('/change_to_false', methods=['POST'])
def change_to_false():
    x = request.form['x']
    y = request.form['y']
    x = eval(x);
    y = eval(y);
    relation = session['relation']
    matrix = np.array(session.get('matrix', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)
    relations = session.get('relations',None)
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    distribution_1 = session.get('distribution_1',None)
    last = session.get('last', None)
    entityX = session.get('entityX',None)
    entityY = session.get('entityY', None)
    operation = session.get('operation', None)
    matrix_ori = np.array(session.get('matrix_ori', None))

    x = entities.index(x)
    y = entities.index(y)
    matrix[x,y,relation] = 0
    matrix_ori[x,y,relation] = 0

    session['matrix'] = matrix.tolist()
    session['matrix_ori'] = matrix_ori.tolist()

    operation[relations[relation]]["mX"].append(x)
    operation[relations[relation]]["mY"].append(y)

    heatmap= create_plot( matrix , entities, relation, distribution_1[relation],entityX, entityY,last,operation[relations[relation]]["mX"],operation[relations[relation]]["mY"],operation[relations[relation]]["qX"],operation[relations[relation]]["qY"])
    session['operation'] = operation

    graphJSON = [heatmap]
    data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)

    return data



@app.route('/change_to_notsure', methods=['POST'])
def change_to_notsure():
    x = request.form['x']
    y = request.form['y']
    x = eval(x);
    y = eval(y);
    relation = session['relation']
    matrix = np.array(session.get('matrix', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)
    relations = session.get('relations',None)
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    last = session.get('last', None)
    distribution_1 = session.get('distribution_1',None)
    # questionX = session.get('questionX', None)
    # questionY = session.get('questionY', None)
    entityX = session.get('entityX',None)
    entityY = session.get('entityY', None)
    # modifyX = session.get('modifyX', None)
    # modifyY = session.get('modifyY', None)
    operation = session.get('operation', None)

    x = entities.index(x)
    y = entities.index(y)

    # session['matrix'] = matrix.tolist()
    operation[relations[relation]]["qX"].append(x)
    operation[relations[relation]]["qY"].append(y)


    heatmap = create_plot( matrix , entities, relation, distribution_1[relation], entityX, entityY,last, operation[relations[relation]]["mX"],operation[relations[relation]]["mY"],operation[relations[relation]]["qX"],operation[relations[relation]]["qY"])
    session['operation'] = operation

    graphJSON = [heatmap]
    data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)

    return data

@app.route('/changecolor', methods=['POST'])
def changecolor():
    num = int(request.form['num'])
    entities = session.get('entities', None)
    entity_embedding = session.get('entity_embedding')
    perp =  session.get('perp')
    session['num'] = num

    line_plot = create_tsne_plot(entity_embedding, entities, num, perp,True)

    graphJSON = [line_plot]
    data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)

    return data

@app.route('/finish', methods=['POST'])
def finish():
    matrix = np.array(session.get('matrix', None))
    matrix_ori = np.array(session.get('matrix_ori', None))
    re_ground_truth = np.array(session.get('re_ground_truth',None))
    entities = session.get('entities', None)

    IDX = np.load("idx_umls_rel.npy")
    AUC_Roc, auc_PR = get_accuracy(re_ground_truth, matrix_ori,IDX,re_ground_truth.shape[0],re_ground_truth.shape[2])
    app.logger.debug("AUC_ROC: %f ; AUC_PR: %f " %(AUC_Roc, auc_PR))

    np.save("final.npy",matrix)

    return

#
# @app.route("/test")
# def test():
#     # do some logic here
#     graphJSON= request.args['graphJSON']
#     relations = session.get('relations',None)
#     return render_template("result_2.html",graphJSON=graphJSON, relations = relations)

if __name__ == '__main__':
    app.run(debug=True)

