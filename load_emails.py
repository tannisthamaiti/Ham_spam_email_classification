import os
import re
import sys
from email import parser as ep
import json
from traceback import print_exc

from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine

from pugnlp import futil


def iter_emails(emaildir='enron_email_files', verbose=True):
    if isinstance(emaildir, list):
        filestats = emaildir
    else:
        if verbose:
            print('Indexing files and folders in {emaildir}'.format(emaildir=emaildir))
        filestats = list(tqdm(futil.generate_files(emaildir)))
    parser = ep.Parser()

    emails = []
    for filestat in tqdm(filestats):
        email = None
        common_encodings = ('utf_8 latin_1 utf_16 shift_jis mac_latin2 iso8859_2 iso8859_3 iso8859_4 iso8859_5 iso8859_6 iso8859_7 iso8859_8 iso8859_9'
                            + 'iso8859_10 iso8859_13 iso8859_14 iso8859_15').split()
        for i, encoding in enumerate(common_encodings):
            try:
                with open(filestat['path'], 'rb') as f:
                    email = parser.parsestr(f.read().decode(encoding))
                break
            except UnicodeDecodeError:
                print('file encoding was not {} for {}... retrying with {}'.format(encoding, filestat['path'], encodings[min(i + 1, len(encodings) - 1)]))
        assert(bool(email))
        email_dict = {}
        for k in vars(email).keys():
            if k == '_headers':
                email_dict.update(dict(email._headers))
            else:
                email_dict[k.strip('_')] = getattr(email, k)
        email_dict['folder'] = os.path.basename(os.path.dirname(filestat['path']))
        email_dict['user'] = os.path.basename(os.path.dirname(os.path.dirname(filestat['path'])))
        yield email_dict


def normalize_colname(name):
    r""" Normalize string column name with .lower(), .strip(), and .replace() on all r'\W' chars

    >>> normalize_colname('Email  Message-ID::')
    email_message_id_
    """
    try:
        return int(float(name))
    except (ValueError, TypeError):
        return re.sub(r'[\W]+', '_', str(name).strip().lower())


def DataFrame(table, *args, **kwargs):
    """ set index or index_col to False to skip automagically guessing the Index column """
    index = kwargs.pop('index', None)
    index_col = kwargs.pop('index_col', None)
    if isinstance(index, str):
        index_col = index
        index = None
    df = pd.DataFrame(table, *args, index=(index or None), **kwargs)
    df.columns = [normalize_colname(c) for c in df.columns]
    if index_col:
        df = df.set_index(normalize_colname(index_col))
    elif index_col is not False and index is not False and len(df) == len(df[df.columns[0]].dropna().unique()):
        df = df.set_index(df.columns[0])
    if df.index.name == 'message_id':
        df.index = df.index.str.lstrip('<').str.rstrip('>').str.strip('.').str.strip("'").str.strip().split('.')
        df.index.name = 'id'  # to be compatible with django
        # df.index = pd.MultiIndex.from_tuples(pd.Series(df.index).apply(tuple), names=['i', 'j', 'server', 'name'])
    return df


def parse_emails(emaildir='/Users/hobsonlane/code/springboard/tannistha/enron_email_files', verbose=True):
    emails = list(iter_emails(emaildir, verbose=verbose))
    if verbose:
        print(emails[:5])
    df = DataFrame(emails, index_col='Message-ID')
    df.fillna('', inplace=True)
    df['defects'] = df['defects'].apply(str)  # otherwise these will be lists and NaNs
    email_users = []  # many-to-many table connecting emails to user email_addresses in the CC, TO, BCC, and FROM fields
    all_users = set()
    for i, e in tqdm(df.iterrows()):
        for c in 'cc bcc from to'.split():
            v = e[c]
            users = set((addr.strip().strip('<').strip('>').strip('.').strip("'") for addr in re.split(r'[\s,]+', v)))
            all_users.update(set(users))
            for u in users:
                if not u:
                    continue
                email_users.append(dict(emailid=i, address=u, kind=c.lower()))
    return df, DataFrame(email_users), DataFrame(sorted(all_users))


def create_table(df, filename='enron_emails.db', name=None, batch_size=200):
    name = name or df.index.name or filename.split('.')[0]
    name = name[:-2] if name.endswith('db') or name.endswith('id') else name
    name = name.strip().strip('.').strip('_').strip('-')
    engine = create_engine('sqlite:///{}'.format(filename), echo=False)
    # large tables, like the emails table will crash sqlite, unless you keep the row batches below 500 or so.
    for i in tqdm(range(0, len(df), batch_size)):
        df.iloc[i:i + batch_size].to_sql(name, con=engine, if_exists='append')


if __name__ == '__main__':
    emaildir = 'enron_email_files'
    if len(sys.argv) > 1:
        emaildir = sys.argv[1]
    emails, email_users, users = parse_emails(emaildir=emaildir)
    emails.fillna('', inplace=True)
    print(emails.head())
    print(emails.shape)
    print(emails.info())
    print(emails.describe())
