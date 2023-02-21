import DNNClassifier
import FeatureEngine
import pandas as pd
import numpy as np



if __name__ == '__main__':
    formula_file='DATA/predict_data.csv'
    feature_eng=FeatureEngine.Features(formula_file=formula_file)
    features=feature_eng.get_features()
    print(np.array(features).shape)

    df=pd.DataFrame(features)
    pred_x= df.iloc[:, 1:].values

    dnn=DNNClassifier.DNN()

    #provide the path to saved model in load_model
    loaded_model, maxm = dnn.load_model('TRAINED/model-2023_02_21_04_30_16.h5')
    print(len(maxm))
    print(pred_x.shape)
    df_mat=pd.read_csv(formula_file,header=None)
    formulas=df_mat.iloc[:,0].values
    dnn.predict(formulas=formulas,model=loaded_model,maxm=maxm,pred_x=pred_x)
