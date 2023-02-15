import numpy as np
import pandas as pd
from keras import models
from keras.layers import Dense
from keras import layers
from keras import optimizers
from keras import *
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
import pickle, os
from datetime import datetime
import FeatureEngine

seed=7
np.random.seed(seed)

class DNN:

    # init method or constructor
    def __init__(self,test_size=0.02):

        #test size of the test set
        self.test_size=test_size

    def create_model(self,num_inputs):
        # model
        model = models.Sequential()
        model.add(layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.0001),
                               input_shape=(num_inputs,)))  # 110
        model.add(Dropout(0.05))  #
        model.add(layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))  # 55
        model.add(Dropout(0.05))  # 0.025
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
        features = feature_eng.get_features(addAVG=True,addAAD=False,addMD=True,addCV=True)
        df=pd.DataFrame(features)
        #df = df.drop([9])
        df = df.dropna()

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(df.values.tolist())
        data_list = imp.transform(df.values.tolist())
        df = pd.DataFrame(data_list)

        return df

    def split_data(self,df):
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

        history = model.fit(train_x, train_y, validation_split=0.1, epochs=1000, batch_size=5000, verbose=0)

        # model evaluation
        results = model.evaluate(test_x, test_y)
        y_rbf_test = (model.predict(test_x) > 0.5).astype(int)
        y_rbf_train = (model.predict(train_x) > 0.5).astype(int)

        return (train_y, y_rbf_train, test_y, y_rbf_test, model)

    def cross_validation(self,data):
        seed = 7

        train_x = data[0]
        train_y = data[2]

        length = len(train_x)
        inc = int(length / 10)

        # model training
        num_inputs = train_x.shape[1]
        model = self.create_model(num_inputs)

        i0 = 0
        i1 = inc

        scores = []

        for i in range(10):
            # print(train_x[i0:i1])
            test_x1 = train_x[i0:i1]
            test_y1 = train_y[i0:i1]
            if (i0 != 0):
                train_x1 = np.array(list(train_x[:i0]) + list(train_x[i1:]))
                train_y1 = np.array(list(train_y[:i0]) + list(train_y[i1:]))
            else:
                train_x1 = train_x[i1:]
                train_y1 = train_y[i1:]
            history = model.fit(train_x1, train_y1, validation_split=0.1, epochs=1000, batch_size=5000, verbose=0)

            # model evaluation
            results = model.evaluate(test_x1, test_y1)
            y_rbf = (model.predict(test_x1) > 0.5).astype(int)
            accuracy = accuracy_score(test_y1, y_rbf)
            i0 = i1
            i1 = i1 + inc

            scores.append(accuracy)

        scores = np.array(scores)
        avg = np.mean(scores)
        std = np.std(scores)
        print(scores)
        print(f'{i} Average accuracy: {avg} +/- {std}')

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
        pickle.dump(model, open(f'{filename}.sav', 'wb'))
        #save the normalizing parameters
        df_maxm = pd.DataFrame(maxm)
        df_maxm.to_csv(f'{dir}/maxm-{now}.csv', index=False, header=None)

    def load_model(self,file_name):
        # load the model from disk
        loaded_model = pickle.load(open(file_name, 'rb'))

        #load normalizing parameters
        dir=os.path.dirname(file_name)
        file_name0=file_name.split(sep='.sav')[0]
        file_name0=file_name0.split(sep='-')[1]

        df_maxm_load = pd.read_csv(f'{dir}/maxm-{file_name0}.csv', header=None)
        maxm = np.array([x[0] for x in df_maxm_load.values.tolist()])

        return loaded_model, maxm

    def predict(self,formulas,model,maxm,pred_x):
        pred_x /= maxm
        y_rbf_pred =model.predict(pred_x)
        y_rbf_pred=list(y_rbf_pred)

        y_pred_label=['metal' if x==0 else 'non-metal' for x in y_rbf_pred]

        dir='RESULTS'
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists(dir):
            os.makedirs(dir)

        results=zip(formulas,y_pred_label)
        df_pred=pd.DataFrame(results)
        df_pred.columns=['formual','class']
        df_pred.to_csv(f'{dir}/dnn_results-{now}.csv',index=False)
