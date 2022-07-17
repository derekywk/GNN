import tensorflow_datasets as tfds
import pandas as pd
import spacy
import numpy as np
from numpy import linalg as LA
import gensim
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from datetime import datetime
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET = "Watches_v1_00"
# DATASET = "Shoes_v1_00"
DATASET_SIZE = -1
DF_FILE_NAME = f"df_{DATASET}_size_{DATASET_SIZE}.pkl.gz"
DF_FILE_NAME_WITH_FEATURES = f"df_{DATASET}_size_{DATASET_SIZE}_with_features.pkl.gz"

GENUINE_THRESHOLD = 0.7
FRAUDULENT_THRESHOLD = 0.3

MODEL_NAME = f"Word2Vec_{DATASET}_size_{DATASET_SIZE}"

def tprint(*args):
    print(datetime.now(), *args)

def wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# WATCH DATASET
# Size of Dataset (960872, 15)
# Genuine Count: 247142 (0.2572059545912463); Fraud Count: 73549 (0.07654401418711337); ratio: 3.360236033120777:1

def keyword_extraction():
    import yake
    custom_kw_extractor = yake.KeywordExtractor(lan='en', n=2, dedupLim=0.9, top=20, features=None)
    keywords = custom_kw_extractor.extract_keywords('\n'.join(df['review_body']))

def pretrained_word_2_vec():
    model = api.load("glove-twitter-25")  # download the model and return as object ready for use
    # model = api.load("fasttext-wiki-news-subwords-300")

def train_word_2_vec_model() -> Word2Vec:
    tprint('Training Word2Vec Model...')
    return Word2Vec(df['processed_words'])

def save_word_2_vec_model(model, name=MODEL_NAME):
    tprint(f'Saving Word2Vec Model as {name}...')
    model.save(name)

def preprocess(col) -> pd.Series:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    words_to_remove = set(stopwords.words('english'))
    words_to_remove.add('')
    word_split_column = col.str.lower().str.split(r"(?<=\w|-)[^\w-]+", regex=True)
    return word_split_column.apply(
        lambda sentence: [
            lemmatizer.lemmatize(word, wordnet_tag(nltk_tag))
            for word, nltk_tag in nltk.pos_tag(sentence)
            if word not in words_to_remove and not word.isnumeric() and len(word) > 1
        ]
    ), word_split_column.str.len()

def keyword_helpful_count(df, keyword, print=False):
    keyword_df = df.loc[df['processed_words_set'].apply(lambda words: keyword in words)]
    genuine_count = keyword_df['genuine'].sum()
    fraud_count = sum(keyword_df['genuine'] == 0)
    if print == 'latex':
        print(fr"{keyword[2:]} & {len(keyword_df)} ({round(len(keyword_df) / len(df) * 100, 2)}\%) & {genuine_count} & {fraud_count} & {round(genuine_count / fraud_count, 2)}:1")
    elif print is True:
        print(f"{{{keyword}}}: Keyword Count: {len(keyword_df)} ({len(keyword_df)/len(df)}); Genuine Count: {genuine_count} ({genuine_count/len(df)}); Fraud Count: {fraud_count} ({fraud_count/len(df)}); ratio: {genuine_count/fraud_count}:1")
    else:
        return len(keyword_df), genuine_count, fraud_count, (genuine_count/fraud_count) if fraud_count else None

def common_words(df, topn=1000):
    counter = Counter(word for words in df['processed_words'] for word in words)
    return counter.most_common(topn)

def important_keywords(df, ratio_threhold=10, topn=50):
    tprint('Extracting Important Keywords...')
    keywords = []
    for word, count in common_words(df):
        keyword_count, genuine_count, fraud_count, ratio = keyword_helpful_count(df, word)
        if ratio is None or ratio > ratio_threhold:
            keywords.append(word)
            if len(keywords) >= topn:
                break
    return keywords

def load_dataset():
    tprint('Loading Dataset...')
    try:
        tprint('Try loading from cache...')
        try:
            df = pd.read_pickle(DF_FILE_NAME_WITH_FEATURES, compression={"method": "gzip", "compresslevel": 1})
            tprint("Successfully loaded DataFrame with features")
        except:
            df = pd.read_pickle(DF_FILE_NAME, compression={"method": "gzip", "compresslevel": 1})
            tprint("Successfully loaded DataFrame without features")
    except:
        tprint('Failed to load from cache...')

        ds, info = tfds.load(f'amazon_us_reviews/{DATASET}', split='train', with_info=True)

        tprint('Creating DataFrame...')
        df = pd.DataFrame(tfds.as_dataframe(ds.take(DATASET_SIZE), info))
        df = df.rename({col: col.replace('data/', '') for col in df.columns}, axis=1)

        tprint('Decoding...')
        for column in df.columns:
            if df[column].dtype == 'O':
                df[column] = df[column].str.decode('utf-8')

        tprint('Preprocessing Words...')
        df['processed_words'], df['word_count'] = preprocess(df["review_body"])

        tprint('Creating Words Sets...')
        df['processed_words_set'] = df['processed_words'].apply(set)

        tprint('Converting Datetime')
        df['review_date'] = df['review_date'].apply(datetime.strptime, args=('%Y-%m-%d',))
        tprint('Finished')

        df['genuine'] = None
        genuine = (df['helpful_votes'] / df['total_votes']) >= GENUINE_THRESHOLD
        fraud = (df['helpful_votes'] / df['total_votes']) <= FRAUDULENT_THRESHOLD
        df.loc[genuine, 'genuine'] = 1
        df.loc[fraud, 'genuine'] = 0

        try:
            df.to_pickle(DF_FILE_NAME, compression={"method": "gzip", "compresslevel": 1})
            tprint("Successfully saved DataFrame")
        except:
            tprint("Failed to save DataFrame")

    tprint('Fetching Target DataFrame...')
    genuine = (df['helpful_votes'] / df['total_votes']) >= GENUINE_THRESHOLD
    fraud = (df['helpful_votes'] / df['total_votes']) <= FRAUDULENT_THRESHOLD
    target_df = df.loc[genuine | fraud]
    genuine_ratio = sum(genuine) / sum(fraud)
    tprint("Size of Dataset", df.shape)
    tprint(f"Genuine Count: {sum(genuine)} ({sum(genuine) / len(target_df)}); Fraud Count: {sum(fraud)} ({sum(fraud) / len(target_df)}); ratio: {genuine_ratio}")

    return df, target_df, genuine_ratio

def normalize_column(col):
    return col/col.max()

def process_behaviour_features(df, type='user', TFIDF=None, W=None):
    assert type in ('user', 'product')
    id_column = 'customer_id' if type == 'user' else 'product_id'
    
    BST_TAU = 28.0
    WRD_ALPHA = 1.5
    ETG_EDGES = [0, 0.5, 1, 4, 7, 13, 33]
    product_avg_rating = df.groupby('product_id')['star_rating'].mean().to_dict()

    if TFIDF is None:
        tprint('Computing TFIDF...')
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words='english')
        TFIDF = vectorizer.fit_transform(df['review_body'])
    else:
        pass # reuse TFIDF

    if W is None:
        tprint('Computing W...')
        uprod, ind_prods = np.unique(df['product_id'], return_inverse=True)
        udates, ind_dates = np.unique([date.toordinal() for date in df['review_date']], return_inverse=True)
        W = np.zeros((len(df),))
        for i in range(len(uprod)):
            ind = ind_prods == i
            if any(ind):
                d = ind_dates[ind]
                if len(d) > 1:
                    ud = np.unique(d)
                    f, edges = np.histogram(d, bins=np.append(ud, ud.max() + 1))
                    m, r = 0, np.zeros((len(d),))
                    for j in range(len(ud)):
                        r[d == ud[j]] = m + 1
                        m += f[j]
                    W[ind] = 1 / (r ** WRD_ALPHA)
                else:
                    r = 1
                    W[ind] = 1 / (r ** WRD_ALPHA)
    else:
        pass  # reuse W

    MNR, PR, NR, avgRD, WRD, BST, ERD, ETG, RL, ACS, MCS = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    tprint(f'Start looping {type} groups...')
    for id, sub_df in df.groupby(id_column, sort=False).__iter__():
        MNR[id] = sub_df.groupby('review_date')['review_id'].count().max()
        PR[id] = sum(sub_df['star_rating'] > 3) / len(sub_df)
        NR[id] = sum(sub_df['star_rating'] < 3) / len(sub_df)
        RD = (sub_df['star_rating'] - sub_df['product_id'].apply(lambda product_id: product_avg_rating[product_id])).abs()
        avgRD[id] = RD.mean()

        datetimes = sub_df['review_date'].sort_values()
        delta_days = (datetimes.iloc[-1] - datetimes.iloc[0]).days

        w = W[sub_df.index]
        WRD[id] = np.dot(RD, w) / np.sum(w)

        if type == 'user': BST[id] = 0.0 if delta_days > BST_TAU else 1.0 - (delta_days / BST_TAU)

        # Code from SOURCE # ETG
        h, _ = np.histogram(sub_df['star_rating'], bins=np.arange(1, 7))
        h = (h[np.nonzero(h)]) / h.sum()
        ERD[id] = (- h * np.log2(h)).sum()

        if len(sub_df) > 1:
            delta_days = [d.days for d in (datetimes.iloc[1:] - datetimes.iloc[:-1].values)]
            delta_days = [d for d in delta_days if d < 33]
            # Code from SOURCE
            h = []
            for delta in delta_days:
                j = 0
                while j < len(ETG_EDGES) and delta > ETG_EDGES[j]:
                    j += 1
                h.append(j)
            _, h = np.unique(h, return_counts=True)
            if h.sum() == 0:
                ETG[id] = 0
            else:
                h = h[np.nonzero(h)] / h.sum()
                ETG[id] = np.sum(- h * np.log2(h))
        else:
            ETG[id] = 0
        ################################################
        
        RL[id] = sub_df['word_count'].mean()
        
        # Code from SOURCE # ACS, MCS
        if len(sub_df) > 1:
            upT = TFIDF[sub_df.index,:]
            npair = int(len(sub_df)*(len(sub_df)-1)/2)
            sim_score = np.zeros((npair,))
            count = 0
            for j in range(len(sub_df)-1):
                for k in range(j+1,len(sub_df)):
                    x, y = upT[j,:].toarray()[0], upT[k,:].toarray()[0]
                    xdoty = np.dot(x,y)
                    if xdoty == 0:
                        sim_score[count] = xdoty
                    else:
                        sim_score[count] = xdoty/(LA.norm(x)*LA.norm(y))
                    count += 1
            ACS[id] = np.mean(sim_score)
            MCS[id] = np.max(sim_score)
        else:
            ACS[id] = -1
            MCS[id] = -1
        ################################################

    tprint('Assigning Features to Columns...')
    df[f'f_{type}_MNR'] = normalize_column(df[id_column].apply(lambda id: MNR[id]))
    df[f'f_{type}_PR'] = df[id_column].apply(lambda id: PR[id])
    df[f'f_{type}_NR'] = df[id_column].apply(lambda id: NR[id])
    df[f'f_{type}_avgRD'] = df[id_column].apply(lambda id: avgRD[id])
    df[f'f_{type}_WRD'] = df[id_column].apply(lambda id: WRD[id])
    if type == 'user': df[f'f_{type}_BST'] = df[id_column].apply(lambda id: BST[id])
    df[f'f_{type}_ERD'] = df[id_column].apply(lambda id: ERD[id])
    df[f'f_{type}_ETG'] = df[id_column].apply(lambda id: ETG[id])
    df[f'f_{type}_RL'] = df[id_column].apply(lambda id: RL[id])
    df[f'f_{type}_ACS'] = df[id_column].apply(lambda id: ACS[id])
    df[f'f_{type}_MCS'] = df[id_column].apply(lambda id: MCS[id])
    
    return TFIDF, W # for reuse

def process_features(df):
    tprint('Processing Features...')
    TFIDF, W = None, None
    if not len([1 for col in df.columns if 'f_user' in col]) == 11:
        tprint('Processing User Features...')
        TFIDF, W = process_behaviour_features(df, type='user')
    if not len([1 for col in df.columns if 'f_type' in col]) == 10:
        tprint('Processing Product Features...')
        TFIDF, W = process_behaviour_features(df, type='product', TFIDF=TFIDF, W=W)
    del TFIDF, W

    try:
        df.to_pickle(DF_FILE_NAME_WITH_FEATURES, compression={"method": "gzip", "compresslevel": 1})
        tprint("Successfully saved DataFrame")
    except:
        tprint("Failed to save DataFrame")

if __name__ == '__main__':
    df, target_df, genuine_ratio = load_dataset()
    process_features(df)
    # print(important_keywords(df, genuine_ratio * 3))
    # model = train_word_2_vec_model()
    # save_word_2_vec_model(model)
    tprint('end')