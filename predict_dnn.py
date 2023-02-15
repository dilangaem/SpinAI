import RFClassifier
import FeatureEngine
import pandas as pd
import numpy as np



if __name__ == '__main__':
    formula_file='DATA/test_materials.csv'
    feature_eng=FeatureEngine.Features(formula_file=formula_file)
    features=feature_eng.get_features()
    print(np.array(features).shape)

    df=pd.DataFrame(features)
    pred_x= df.iloc[:, 1:].values

    rfc=RFClassifier.RFC()

    #provide the path to saved model in load_model
    loaded_model, maxm = rfc.load_model('TRAINED/model-2023_02_15_00_28_17.sav')
    print(len(maxm))
    print(pred_x.shape)
    df_mat=pd.read_csv(formula_file,header=None)
    formulas=df_mat.iloc[:,0].values
    rfc.predict(formulas=formulas,model=loaded_model,maxm=maxm,pred_x=pred_x)
