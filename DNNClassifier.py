import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
import keras
from keras import models
from keras.layers import Dense
from keras import layers
from keras import optimizers
from keras import *
from keras.layers import Dropout
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
import pickle, os
from datetime import datetime
import FeatureEngine

seed = 7
np.random.seed(seed)
rn.seed(10)
tf.random.set_seed(seed)

class DNN:

    # init method or constructor
    def __init__(self,test_size=0.02):

        #test size of the test set
        self.test_size=test_size

    def create_model(self,num_inputs):
        # model
        model = models.Sequential()
        model.add(layers.Dense(56, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                               input_shape=(num_inputs,)))  # 110
        model.add(Dropout(0.5))  #
        model.add(layers.Dense(28, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))  # 55
        model.add(Dropout(0.5))  # 0.025
        # model.add(layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))#22
        # model.add(Dropout(0.05))#0.025
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def load_data(self,file_name='DATA/data_file.csv'):
        '''
        df = pd.read_csv(file_name, header=None)
        '''
        feature_eng = FeatureEngine.Features(formula_file=file_name)
        features = feature_eng.get_features(addAVG=True,addAAD=False,addMD=True,addCV=False)
        df=pd.DataFrame(features)
        #df = df.drop([9])
        df = df.dropna()

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(df.values.tolist())
        data_list = imp.transform(df.values.tolist())
        df = pd.DataFrame(data_list)

        return df

    def split_data(self,df):
        df0 = df.loc[df[0] == 0]
        df1 = df.loc[df[0] == 1]
        row_list = []
        for indx, row in df0.iterrows():
            row_list.append(row)

        for indx, row in df1.iterrows():
            row_list.append(row)

        df = pd.DataFrame(row_list)
        data_X, data_y = df.iloc[:, 1:].values, df.iloc[:, 0].values

        train_x, test_x, train_y, test_y = train_test_split(data_X, data_y, test_size=self.test_size, random_state=42)

        return (train_x, test_x, train_y, test_y)

    def normalize_data(self,data):
        train_x, test_x, train_y, test_y =data

        print(f'Train_x set shape: {train_x.shape}')
        print(f'Train_y set shape: {train_y.shape}')

        print(f'Test_x set shape: {test_x.shape}')
        print(f'Test_y set shape: {test_y.shape}')

        xx = abs(train_x)
        maxm = xx.max(axis=0)
        maxm[maxm == 0.0] = 1

        train_x /= maxm
        test_x /= maxm

        return (train_x, test_x, train_y, test_y,maxm)

    def run_ml(self,data):
        print('---------- Training the Model ------------')
        train_x, test_x, train_y, test_y, maxm = data

        # model training
        num_inputs=train_x.shape[1]
        model = self.create_model(num_inputs)
        validation_split = 0.1666
        epochs = 500
        batch_size = 1500
        history = model.fit(train_x, train_y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=0)

        # cross Validation
        print('*************************************************************************')
        print('Cross Validation')
        num_folds = 3
        self.cross_validation(train_x, train_y, num_folds=num_folds, validation_split=validation_split, epochs=epochs, batch_size=batch_size)

        # model evaluation
        results = model.evaluate(test_x, test_y)
        y_rbf_test = (model.predict(test_x) > 0.5).astype(int)
        y_rbf_train = (model.predict(train_x) > 0.5).astype(int)

        return (train_y, y_rbf_train, test_y, y_rbf_test, model)

    def cross_validation(self,train_x, train_y, num_folds,epochs, batch_size, validation_split):
        kfold = KFold(n_splits=num_folds, shuffle=True)
        n = 1
        scores_list = []
        for train, valid in kfold.split(train_x, train_y):
            # print(f'--------Validation {n}---------')
            num_inputs = train_x.shape[1]
            modelK = self.create_model(num_inputs)
            history = modelK.fit(train_x[train], train_y[train], validation_split=validation_split, epochs=epochs,
                                 batch_size=batch_size, verbose=0)
            scores = modelK.evaluate(train_x[valid], train_y[valid], verbose=0)
            print(f'Fold number {n}: {modelK.metrics_names[1]}={scores[1]}')
            scores_list.append(scores[1])
            n += 1
        print(f'Mean {modelK.metrics_names[1]}: {np.average(scores_list)}+/-{np.std(scores_list)}')

    def print_clf_report(self,results):
        train_y, y_rbf_train, test_y, y_rbf_test, model =results
        print(f'Confusion Matrix: {confusion_matrix(test_y, y_rbf_test)}')
        print('The Classification Report')
        print(classification_report(test_y, y_rbf_test))

    def save(self,model,maxm):
        dir='TRAINED'
        now=datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        filename = f'{dir}/model-{now}'
        if not os.path.exists(dir):
            os.makedirs(dir)


        # save the model to disk
        model.save(f'{filename}.h5')
        #save the normalizing parameters
        df_maxm = pd.DataFrame(maxm)
        df_maxm.to_csv(f'{dir}/maxm-{now}.csv', index=False, header=None)

    def load_model(self,file_name):
        # load the model from disk
        loaded_model = models.load_model(file_name)

        #load normalizing parameters
        dir=os.path.dirname(file_name)
        file_name0=file_name.split(sep='.h5')[0]
        file_name0=file_name0.split(sep='-')[1]

        df_maxm_load = pd.read_csv(f'{dir}/maxm-{file_name0}.csv', header=None)
        maxm = np.array([x[0] for x in df_maxm_load.values.tolist()])

        return loaded_model, maxm

    def predict(self,formulas,model,maxm,pred_x):
        pred_x /= maxm
        y_rbf_pred =(model.predict(pred_x) > 0.5).astype(int)
        y_rbf_pred=list(y_rbf_pred)

        y_pred_label=['magnetic compound' if x==0 else 'nonmagnetic compound' for x in y_rbf_pred]

        dir='RESULTS'
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists(dir):
            os.makedirs(dir)

        results=zip(formulas,y_pred_label)
        df_pred=pd.DataFrame(results)
        df_pred.columns=['formual','class']
        df_pred.to_csv(f'{dir}/dnn_results-{now}.csv',index=False)
