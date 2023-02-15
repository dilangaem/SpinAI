import  DNNClassifier
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    #parser.add_argument('--sym', type=str, required=True)
    parser.add_argument('--test_size', type=float, required=True)
    args = parser.parse_args()


    rfc = DNNClassifier.DNN(test_size=args.test_size)
    df = rfc.load_data(file_name=f'DATA/{args.file_name}')
    data = rfc.split_data(df)
    norm_data = rfc.normalize_data(data)
    results = rfc.run_ml(norm_data)
    rfc.print_clf_report(results)

    model = results[4]
    maxm = norm_data[4]
    rfc.save(model, maxm)