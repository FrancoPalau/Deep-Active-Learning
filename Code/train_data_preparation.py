import argparse
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

from dga_tokenizer import tokenize, tokenize_no_padding, tokenize_with_padding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training data preparation script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--csv-file", help="CSV file with domains and labels")
    parser.add_argument("--test-only", action="store_true",
                        help="Bypass training and test split")
    parser.add_argument("--maxlen", type=int,
                        help="Size of string to consider")    
    args = parser.parse_args()
    df = pd.read_csv(args.csv_file,
                     header=0,
                     names=['domain', 'class'],
                     dtype={'domain': str, 'class': np.int8})
    #df = df.drop_duplicates('domain')
    maxlen = args.maxlen
    y = df['class'].values
    x = tokenize(df['domain'].values, maxlen)
    #print(len(x))

    if args.test_only:
        pickle.dump((x, y), open('test_data_only.pkl', 'wb'))       
        sys.exit(0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)#
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, stratify=y_test, test_size=0.99)#

    
    pickle.dump((x_train, y_train), open('train_data.pkl', 'wb'))
    pickle.dump((x_valid, y_valid), open('valid_data.pkl', 'wb'))
    pickle.dump((x_test, y_test), open('test_data.pkl', 'wb'))
    #pickle.dump((x, y), open('test_paper_tunnel.pkl', 'wb'))
