# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:35:04 2018

@author: Administrator
"""
import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate_sade import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve




#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run ANMF.")
    parser.add_argument('--path', nargs='?', default='Data/DR/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='dr',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=256,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=10,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, dtype=None):
    return initializers.Zeros()


def get_model(u_dim, i_dim, latent_dim, regs=[0,0]):
    # user auto-encoder
    u_input = Input(shape=(u_dim,), dtype='float32', name='u_input')
    u_input_c = Input(shape=(313,), dtype='float32', name='u_input_c')
    
    u_encoded = Dense(latent_dim, activation='linear')(u_input)
    u_encoded_c = Dense(latent_dim, activation='linear')(u_input_c)
    x = keras.layers.concatenate([u_encoded,u_encoded_c])
    #u_encoded = Dense(64, activation='relu')(u_encoded)
    u_middle = Dense(latent_dim, activation='relu', name='u_middle')(x)
    #u_decoded = Dense(64, activation='relu')(u_middle)
    #u_decoded = Dense(128, activation='relu')(u_decoded)
    u_decoded = Dense(u_dim, activation='sigmoid')(u_middle)
    u_decoded_c = Dense(313, activation='sigmoid')(u_middle)

    u_autoencoder = Model([u_input,u_input_c], [u_decoded,u_decoded_c])

    # item auto-encoder
    i_input = Input(shape=(i_dim,), dtype='float32', name='i_input')
    i_input_c = Input(shape=(593,), dtype='float32', name='i_input_c')
    #i_encoded = Dense(64, activation='relu')(i_input)
    i_encoded=Dense(latent_dim, activation='linear')(i_input)
    i_encoded_c=Dense(latent_dim, activation='linear')(i_input_c)
    x = keras.layers.concatenate([i_encoded,i_encoded_c])
    i_middle = Dense(latent_dim, activation='relu', name='i_middle')(x)
    #i_decoded = Dense(64, activation='relu')(i_middle)
    i_decoded = Dense(i_dim, activation='sigmoid')(i_middle)
    i_decoded_c = Dense(593, activation='sigmoid')(i_middle)

    i_autoencoder = Model([i_input,i_input_c], [i_decoded,i_decoded_c])



    predict_vector = keras.layers.Multiply()([u_middle, i_middle])
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)
    hybrid_model = Model(input=[u_input, u_input_c, i_input, i_input_c], output=prediction)
    return hybrid_model


def get_train_instances(train, num_negatives, uSimMat, iSimMat,DiDrAMat,neg_sample):
    user_input, item_input, labels = [], [], []
    user_input_c,item_input_c=[],[]

    for index in range(len(train)):
        # positive instance
        instance=train[index]
        u=int(instance[0])
        i=int(instance[1])
        user_input.append(uSimMat[u])
        item_input.append(iSimMat[i])
        user_input_c.append(DiDrAMat[:,u])
        item_input_c.append(DiDrAMat[i])
        labels.append(1)
        #negative instances
        for t in range(num_negatives):
            j = np.random.randint(len(neg_sample[u]))
            ins=neg_sample[u][j]
            user_input.append(uSimMat[u])
            item_input.append(iSimMat[ins])
            user_input_c.append(DiDrAMat[:,u])
            item_input_c.append(DiDrAMat[ins])
            labels.append(0)
      
    return user_input, item_input, labels,user_input_c,item_input_c

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot( thresholds, precisions[:-1], "b--", label="Precision" )
    plt.plot( thresholds, recalls[:-1], "g-", label="Recall" )
    plt.xlabel( "Threshold" )
    plt.legend(loc="upper left")
    plt.ylim( [0, 1] )
    plt.show()

def plot_roc_curve( fpr, tpr, label=None):
    plt.plot( fpr, tpr, linewidth=2, label=label )
    plt.plot( [0,1], [0,1], "k--" )
    plt.axis([0,1,0,1])
    plt.xlabel( "False Positive Rate" )
    plt.ylabel( "True Positive Rate" )


if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("GMF arguments: %s" %(args))
    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' %(args.dataset, num_factors, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, uSimMat, iSimMat,DiDrAMat,neg_sample = dataset.trainMatrix, dataset.testRatings, \
                                                          dataset.uSimMat, dataset.iSimMat, dataset.DiDrAMat,dataset.Sim_order
    #num_users, num_items = train.shape
    
    #print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
         # %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(593, 313, num_factors, regs)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    #print(model.summary())
    
    # Init performance
    t1 = time()
    hits,a,fp, tp, th,area= evaluate_model(model, testRatings, uSimMat, iSimMat, DiDrAMat, topK, evaluation_threads,train)

    # Train model

    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels,user_input_c,item_input_c = get_train_instances(train, num_negatives, uSimMat, iSimMat,DiDrAMat,neg_sample)
        
        # Training
        hist = model.fit([np.array(user_input),np.array(user_input_c), np.array(item_input),np.array(item_input_c)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            hit,auc,fpr, tpr, thre,area_pr= evaluate_model(model, testRatings, uSimMat, iSimMat, DiDrAMat, topK, evaluation_threads,train)
            hr, loss = np.array(hits).mean(), hist.history['loss'][0]

           
            print('area_pr的值为'+str(area_pr))           
            print('auc的值为'+str(auc))
            print('hit的值为'+str(hit))
            #plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            if hit> hits:
                hits=hit
            if auc>a:
                a=auc
    print("End. Best Iteration %d:  HR = %.4f. " %(hits,a))


