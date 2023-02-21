import  DNNClassifier
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    #parser.add_argument('--sym', type=str, required=True)
    parser.add_argument('--test_size', type=float, required=True)
    args = parser.parse_args()


    dnn = DNNClassifier.DNN(test_size=args.test_size)
    df = dnn.load_data(file_name=f'DATA/{args.file_name}')

    data = dnn.split_data(df)
    norm_data = dnn.normalize_data(data)
    results = dnn.run_ml(norm_data)
    dnn.print_clf_report(results)

    model = results[4]
    maxm = norm_data[4]
    dnn.save(model, maxm)