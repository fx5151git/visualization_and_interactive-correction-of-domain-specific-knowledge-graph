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
from sklearn.metrics import precision_recall_curve, auc
from sklearn.cluster import KMeans
from rescal import rescal_als
# import matplotlib.pyplot as plt
from bisect import bisect
from flask_session import Session


app = Flask(__name__)
app.secret_key = 'why would I tell you my secret key?'

app.config['SESSION_TYPE'] = 'filesystem'
TEMPLATES_AUTO_RELOAD = True

sess = Session()
sess.init_app(app)

def predict_rescal_als(T):
    A, R, f, itr, exectimes = rescal_als(
        T, 120, init='nvecs',
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
    return entity_embedding, P, R, auc(recall, prec)


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

def generate_data():

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
    e, k = K.shape[0], K.shape[2]
    SZ = e * e * k

    # copy ground truth before preprocessing
    GROUND_TRUTH = K.copy()

    # construct array for rescal
    T = [lil_matrix(K[:, :, i]) for i in range(k)]

    # Do relation cross-validation
    Percentage = 5
    ID_relation = list(range(k))
    ID_relation.pop(5)
    shuffle(ID_relation)
    IDX = list(range(SZ))
    rel_percentage = int(e * e / Percentage)

    frelation = 4
    offset = 0
    relation_val = []

    idx_test = []
    idx_relation = ID_relation[:frelation]

    relation_val.append(ID_relation[offset:offset + frelation])
    for rel in idx_relation:
        print(relations[rel])
        random_list = IDX[rel * e * e:(rel + 1) * e * e]
        shuffle(random_list)
        idx_test = idx_test + random_list[:rel_percentage]

    entity_embedding, tensor, rel_embedding, auc = innerfold(T, idx_test, IDX, e, k, SZ, GROUND_TRUTH)
    bar_index, bar_value, distribution_0_visualize = get_outlier_scores(tensor, GROUND_TRUTH, e, k, idx_test)

    for m in range(tensor.shape[2]):
        for n in range(tensor.shape[0]):
            for t in range(tensor.shape[1]):
                if(tensor[n,t,m]<0):
                    tensor[n,t,m] = 0
    # 12
    cluster = 12
    re_tensor, entities, last, re_Ground_truth = entity_cluster_tensor(cluster,entity_embedding, tensor, entities, GROUND_TRUTH)

    return re_tensor, bar_index, bar_value, entities, relations, GROUND_TRUTH, distribution_0_visualize, last, re_Ground_truth

def entity_cluster_tensor(clusters, embedding, tensor,entities, ground_truth):
    # kmeans for error cluster
    # color = np.random.rand(clusters)
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(embedding)

    labels = kmeans.labels_

    index = []
    last = []
    for i in range(clusters):
        indices = [m for m, x in enumerate(labels) if x == i]
        last.append(indices[-1])
        index = index + indices

    entities = [entities[i] for i in index]
    result = tensor.copy()
    re_ground_truth = ground_truth.copy()
    for i in range(tensor.shape[2]):
        result[:, :, i] = tensor[index, :, i]
        result[:, :, i] = result[:, index, i]
        re_ground_truth[:, :, i] = ground_truth[index, :, i]
        re_ground_truth[:, :, i] = re_ground_truth[:, index, i]

    return result, entities, last, re_ground_truth


def create_plot(GROUND_TRUTH, P , entities, relation, distribution_0, entityX, entityY, last, update, modifyX, modifyY):

    heat_data = []


    for i in range(len(entities)):
        new_row = []
        for j in range(len(entities)):
            new_row.append(P[j, i, relation])
        heat_data.append(list(new_row))


    data = [go.Heatmap(
            x=entities,
            y=entities,
            z=heat_data,
            xgap=1,
            ygap=1,
            colorscale=[
                [0, "rgb(242,240,247)"],
                [distribution_0/4,"rgb(242,240,247)"],

                [distribution_0/4,"rgb(203,201,226)"],
                [distribution_0/2, "rgb(203,201,226)"],

                [distribution_0 / 2, "rgb(158,154,200)"],
                [distribution_0/4*3, "rgb(158,154,200)"],

                [distribution_0 / 4 * 3, "rgb(117,107,177)"],
                [distribution_0, "rgb(117,107,177)"],

                [distribution_0, "rgb(84,39,143)"],
                [1, "rgb(84,39,143)"],

        ])]


    fig = go.Figure(data=data)

    fig.add_trace(
        go.Scatter(
            x=entityX,
            y=entityY,
            mode="markers",
            hoverinfo='skip',
            marker=dict(size=4.5,
                        color=0),
            showlegend=False)
    )

    fig['layout'].update({
        # 'autosize': True,
        'width': 900, 'height': 900,
                             'margin':dict(
                                        l=230,
                                        r=0,
                                        b=230,
                                        t=0,
                                        pad=4
                                    ),
                             'showlegend': False,
                             'hovermode': 'closest',
                             'xaxis' : dict(tickfont= dict(
                                                size=8,
                                                color='black'
                                            ),
                                            tickangle = 40),
                             'yaxis': dict(tickfont=dict(
                                                size=8,
                                                color='black'
                                            )),
                            'shapes' :[
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[0],
                                    x1=entities[-1],
                                    y1=last[0],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[1],
                                    x1=entities[-1],
                                    y1=last[1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[2],
                                    x1=entities[-1],
                                    y1=last[2],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[3],
                                    x1=entities[-1],
                                    y1=last[3],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[4],
                                    x1=entities[-1],
                                    y1=last[4],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[5],
                                    x1=entities[-1],
                                    y1=last[5],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[6],
                                    x1=entities[-1],
                                    y1=last[6],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[7],
                                    x1=entities[-1],
                                    y1=last[7],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[8],
                                    x1=entities[-1],
                                    y1=last[8],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[9],
                                    x1=entities[-1],
                                    y1=last[9],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=entities[0],
                                    y0=last[10],
                                    x1=entities[-1],
                                    y1=last[10],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[0],
                                    y0=entities[0],
                                    x1=last[0],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[1],
                                    y0=entities[0],
                                    x1=last[1],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[2],
                                    y0=entities[0],
                                    x1=last[2],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[3],
                                    y0=entities[0],
                                    x1=last[3],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[4],
                                    y0=entities[0],
                                    x1=last[4],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[5],
                                    y0=entities[0],
                                    x1=last[5],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[6],
                                    y0=entities[0],
                                    x1=last[6],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[7],
                                    y0=entities[0],
                                    x1=last[7],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[8],
                                    y0=entities[0],
                                    x1=last[8],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[9],
                                    y0=entities[0],
                                    x1=last[9],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                ),
                                go.layout.Shape(
                                    type="line",
                                    x0=last[10],
                                    y0=entities[0],
                                    x1=last[10],
                                    y1=entities[-1],
                                    opacity=0.3,
                                    line=dict(
                                        color="Crimson",
                                        width=1,
                                    )
                                )
                            ]
                             })

    if update == 1 :
        for i, j in zip(modifyX,modifyY):
            fig['layout'].update({
                'images': [
                    go.layout.Image(
                        source="static/pen.png",
                        xref="x",
                        yref="y",
                        x=i,
                        y=j,
                        sizex=0.3,
                        sizey=0.3,
                        sizing="stretch",
                        opacity=0.5,
                        layer="above")
                ]
        })

        if update == 2:
            for i, j in zip(modifyX, modifyY):
                fig['layout'].update({
                    'images': [
                        go.layout.Image(
                            source="static/questionmark.png",
                            xref="x",
                            yref="y",
                            x=i,
                            y=j,
                            sizex=0.3,
                            sizey=0.3,
                            sizing="stretch",
                            opacity=0.5,
                            layer="above")
                    ]
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

def get_true_entity(original, relation, entities):
    entityX = []
    entityY = []

    for n in range(original.shape[0]):
        for t in range(original.shape[1]):
            if(original[n,t,relation] == 1):
                entityX.append(entities[n])
                entityY.append(entities[t])

    return entityX,entityY





@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    modifyX = []
    modifyY = []
    update = 0

    session['questionX'] = []
    session['questionY'] = []

    matrix, bar_index, bar_value, entities, relations, GROUND_TRUTH, distribution_1, last, re_ground_truth = generate_data()
    bar, relation_chosen = create_bar_plot(relations, bar_index, bar_value)
    relation = relations.index(relation_chosen[-1])
    session['relation'] = relation
    entityX, entityY = get_true_entity(re_ground_truth, relation, entities)
    heatmap = create_plot(GROUND_TRUTH, matrix, entities, relation, distribution_1[relation], entityX, entityY, last, update, modifyX, modifyY)
    graphJSON = [heatmap, bar]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)


    session['matrix'] = matrix.tolist()
    session['bar_index'] = bar_index
    session['bar_value'] = bar_value
    session['relation_chosen'] = relation_chosen
    session['entities'] = entities
    session['relations'] = relations
    session['Ground_truth'] = GROUND_TRUTH.tolist()
    session['distribution_1'] =  distribution_1
    session['last'] = last
    session['re_ground_truth'] = re_ground_truth.tolist()
    session['modifyX'] = modifyX
    session['modifyY'] = modifyY

    session['entityX'] = entityX
    session['entityY'] = entityY

    return render_template('result_2.html', graphJSON=graphJSON, relations=relations, relation=relation)


@app.route('/configuration', methods=['POST'])
def configuration():
    relation = request.form['relation']
    matrix = np.array(session.get('matrix', None))
    bar_index = session.get('bar_index', None)
    bar_value = session.get('bar_value' , None)
    entities = session.get('entities', None)
    relations = session.get('relations',None)
    GROUND_TRUTH = np.array(session.get('Ground_truth',None))
    distribution_1 = session.get('distribution_1',None)
    last = session.get('last',None)
    re_ground_truth = np.array(session.get('re_ground_truth',None))

    relation = relations.index(relation)
    session['relation'] = relation

    entityX, entityY = get_true_entity(re_ground_truth, session['relation'],entities)

    session['entityX'] = entityX
    session['entityY'] = entityY

    update = 0
    modifyX = session.get('modifyX', None)
    modifyY = session.get('modifyY', None)

    heatmap = create_plot(GROUND_TRUTH, matrix, entities, relation, distribution_1[relation],entityX, entityY, last, update, modifyX, modifyY)
    bar = create_bar_plot(relations,bar_index, bar_value)
    graphJSON = [heatmap, bar]
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('result_2.html', graphJSON=graphJSON, relations = relations, relation = relation)

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

    relation = relations.index(relation_chosen[relation])
    session['relation'] = relation

    entityX, entityY = get_true_entity(re_ground_truth, session['relation'], entities)

    session['entityX'] = entityX
    session['entityY'] = entityY

    update = 0
    modifyX = session.get('modifyX', None)
    modifyY = session.get('modifyY', None)

    heatmap = create_plot(GROUND_TRUTH, matrix , entities, relation, distribution_1[relation], entityX, entityY, last, update, modifyX, modifyY)
    bar = create_bar_plot(relations,bar_index, bar_value)
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
    entityX = session.get('entityX',None)
    entityY = session.get('entityY', None)
    entityX.append(x)
    entityY.append(y)

    x = entities.index(x)
    y = entities.index(y)

    session['matrix'] = matrix.tolist()

    update = 1
    modifyX = session.get('modifyX', None)
    modifyY = session.get('modifyY', None)
    modifyX.append(x)
    modifyY.append(y)

    heatmap = create_plot(GROUND_TRUTH, matrix , entities, relation, distribution_1[relation], entityX, entityY,last, update, modifyX, modifyY)
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

    x = entities.index(x)
    y = entities.index(y)
    matrix[x,y,relation] = 0

    session['matrix'] = matrix.tolist()

    update = 1
    modifyX = session.get('modifyX', None)
    modifyY = session.get('modifyY', None)
    modifyX.append(x)
    modifyY.append(y)

    heatmap = create_plot(GROUND_TRUTH, matrix , entities, relation, distribution_1[relation],entityX, entityY,last, update, modifyX, modifyY)
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
    questionX = session.get('questionX', None)
    questionY = session.get('questionY', None)
    entityX = session.get('entityX',None)
    entityY = session.get('entityY', None)
    entityX.append(x)
    entityY.append(y)

    x = entities.index(x)
    y = entities.index(y)

    session['matrix'] = matrix.tolist()

    update = 2
    modifyX = session.get('modifyX', None)
    modifyY = session.get('modifyY', None)
    modifyX.append(x)
    modifyY.append(y)

    heatmap = create_plot(GROUND_TRUTH, matrix , entities, relation, distribution_1[relation], entityX, entityY,last, update, questionX, questionY)
    graphJSON = [heatmap]
    data = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)

    return data



#
# @app.route("/test")
# def test():
#     # do some logic here
#     graphJSON= request.args['graphJSON']
#     relations = session.get('relations',None)
#     return render_template("result_2.html",graphJSON=graphJSON, relations = relations)



if __name__ == '__main__':
    app.run(debug=True)

