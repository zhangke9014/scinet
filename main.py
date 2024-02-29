from data_load import load_data
from train_model import train
import argparse


def parser_initial():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataPath', type=str, default='data.xlsx')
    parser.add_argument('--labelPath', type=str, default='label.xlsx')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_initial()
    trainData, testData, trainLabel, testLabel = load_data(args.dataPath, args.labelPath)
    y = train(trainData, trainLabel)
    print('~')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
