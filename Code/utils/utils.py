import string
from sklearn.model_selection import train_test_split
import csv
import sqlite3


def split_data(x, y, stratify=None, split_pct=0.1):
    """
    Splits data into training and testing.
    """
    return train_test_split(x, y, stratify=stratify,test_size=split_pct)


def get_data_from_file(filename):
    with open(filename, 'r') as data_file:
        # return [sample.rsplit('.', 1)[0] for sample in data_file]
        return [sample.strip() for sample in data_file]


def text_filter():
    f = '\n'
    return f


def get_valid_characters():
    """
        Returns a string with all the valid characters
    :return:
    """
    # '$' acts as the masking character
    return '$' + 'abcdefghiABCDEFGHIrstuvwxyzRSTUVWXYZ0123456789.,+*'


def split_with_spaces(data, nb_chars=1):
    """
        Splits every string in data with spaces every nb_chars.

    :param data:
    :param nb_chars:
    :return: list(str)
    """
    n = nb_chars
    text = []
    for x in data:
        text.append(" ".join([x[i:i + n] for i in range(0, len(x), n)]))
    assert len(text) == len(data)
    return text


def to_csv(filename, data, columns):
    with open(filename, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(columns)
        csv_out.writerows(data)


def to_sqlite(filename, data):
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE results
                 (sample TEXT, label TEXT, class INTEGER, predicted_class INTEGER, predicted_probability REAL)''')
    # Create label index
    c.execute('''CREATE INDEX label_index ON results (label)''')
    # Insert a row of data
    c.executemany('insert into results values (?,?,?,?)', data)
    # Save (commit) the changes
    conn.commit()
    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()
