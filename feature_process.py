import tensorflow_datasets as tfds
import pandas as pd
import spacy
import numpy as np
import math
from numpy import linalg as LA
from scipy import sparse
import gensim
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from datetime import datetime
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import tprint, is_ascii
import json

DATASET = "Watches_v1_00"
# DATASET = "Shoes_v1_00"
DATASET_SIZE = -1 # -1 refers to whole dataset
DF_FILE_NAME = f"df_{DATASET}_size_{DATASET_SIZE}.pkl.gz"
DF_FILE_NAME_WITH_FEATURES = f"df_{DATASET}_size_{DATASET_SIZE}_with_features.pkl.gz"
KEYWORDS_SIZE = 200
KEYWORDS_FILE_NAME = f"KEYWORD_{DATASET}_size_{DATASET_SIZE}_kw_size_{KEYWORDS_SIZE}.json"
GIST_FEATURES_STATS_FILE_NAME = "GIST_FEATURES_STATS.json"

GENUINE_THRESHOLD = 0.7
FRAUDULENT_THRESHOLD = 0.3

# WORD_2_VEC_MODEL_NAME = f"Word2Vec_{DATASET}_size_{DATASET_SIZE}"
WORD_2_VEC_MODEL_NAME = f"Word2Vec_{DATASET}_size_{-1}"

def wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a' # wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return 'v' # wordnet.VERB
    elif nltk_tag.startswith('N'):
        return 'n' # wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return 'r' # wordnet.ADV
    else:
        return 'n' # wordnet.NOUN

def pretrained_word_2_vec():
    model = api.load("glove-twitter-25")  # download the model and return as object ready for use
    # model = api.load("fasttext-wiki-news-subwords-300")

def train_word_2_vec_model(df) -> Word2Vec:
    tprint('Training Word2Vec Model...')
    return Word2Vec(df['processed_words'])

def save_word_2_vec_model(model, name=WORD_2_VEC_MODEL_NAME):
    tprint(f'Saving Word2Vec Model as {name}...')
    model.save(name)

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

def important_keywords(df, ratio_threhold=10, topn=KEYWORDS_SIZE):
    try:
        with open(KEYWORDS_FILE_NAME, "r") as file:
            keywords = json.load(file)
        tprint('Successfully loaded Important Keywords...')
    except:
        tprint('Extracting Important Keywords...')
        keywords = []
        for word, count in common_words(df, topn*20):
            keyword_count, genuine_count, fraud_count, ratio = keyword_helpful_count(df, word)
            if (ratio is None or ratio > ratio_threhold) and len(word) > 2 and word.isalpha():
                keywords.append(word)
                if len(keywords) >= topn:
                    break
        with open(KEYWORDS_FILE_NAME, "w") as file:
            json.dump(keywords, file)
    return keywords

def TFIDF(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(lowercase=False)
    tfIdf = vectorizer.fit_transform(df['processed_words'].apply(' '.join))
    result = pd.DataFrame(tfIdf[0].T.todense(), index=vectorizer.get_feature_names_out(), columns=["TF-IDF"]).sort_values('TF-IDF', ascending=False)

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
        df["review_body"] = df["review_body"].str.replace('<br />', ' ')
        df['processed_words'], df['word_list'] = preprocess(df["review_body"])

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
    genuine_valid = genuine & (df['total_votes'] > 1)
    fraud_valid = fraud & (df['total_votes'] > 1)
    df_polar = df.loc[genuine | fraud]
    df_valid = df.loc[genuine_valid | fraud_valid]
    genuine_ratio = sum(genuine) / sum(fraud)
    tprint("Shape of Dataset:", df.shape)
    print("Whole:", len(df))
    print("Polar:", len(df_polar))
    print("Valid:", len(df_valid))
    print(f"Genuine Count: {sum(genuine)} ({sum(genuine) / len(df_polar)}); Fraud Count: {sum(fraud)} ({sum(fraud) / len(df_polar)}); ratio: {genuine_ratio}")

    return df, df_polar, df_valid, genuine_ratio

def normalize_column(col):
    return col/col.max()

def preprocess(col) -> pd.Series:
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    words_to_remove = set(stopwords.words('english'))
    word_split_column = col.str.split(r"(?<=\w|-)[^\w-]+", regex=True).apply(lambda sentence: [word for word in sentence if len(word) > 0])
    return word_split_column.apply(
        lambda sentence: [
            lemmatizer.lemmatize(word, wordnet_tag(nltk_tag))
            for word, nltk_tag in nltk.pos_tag([word.lower() for word in sentence])
            if word not in words_to_remove and not word.isnumeric() and len(word) > 1
        ]
    ), word_split_column

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
        RL[id] = sub_df['word_list'].str.len().mean()
        datetimes = sub_df['review_date'].sort_values()
        if type == 'user':
            delta_days = (datetimes.iloc[-1] - datetimes.iloc[0]).days
            BST[id] = 0.0 if delta_days > BST_TAU else 1.0 - (delta_days / BST_TAU)
        RD = (sub_df['star_rating'] - sub_df['product_id'].apply(lambda product_id: product_avg_rating[product_id])).abs()
        avgRD[id] = RD.mean()
        w = W[sub_df.index]
        WRD[id] = np.dot(RD, w) / np.sum(w)

        ################################################
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

def process_review_behaviour_features(df):
    DEV_BETA = 0.63
    ETF_DELTA, ETF_BETA = 7 * 30, 0.69

    int_dates = np.array([date.toordinal() for date in df['review_date']])
    udates, ind_dates = np.unique(int_dates, return_inverse=True)
    uuser, ind_users = np.unique(df['customer_id'], return_inverse=True)
    uprod, ind_prods = np.unique(df['product_id'], return_inverse=True)
    ratings = df['star_rating'].to_numpy()

    ################################################
    tprint('Computing RANK...')
    Rank = np.zeros((len(df),))
    for i in range(len(uprod)):
        ind = ind_prods==i
        if any(ind):
            d = ind_dates[ind]
            if len(d)>1:
                ud = np.unique(d)
                f, edges = np.histogram(d, bins=np.append(ud,ud.max()+1))
                m, r = 0, np.zeros((len(d),))
                for j in range(len(ud)):
                    r[d==ud[j]] = m + 1
                    m += f[j]
                Rank[ind] = r
            else:
                r = 1
                Rank[ind] = r
    ################################################
    tprint('Computing RD...')
    avgRating = np.zeros((len(df),))
    for i in range(len(uprod)):
        ind = ind_prods==i
        r = ratings[ind]
        avgRating[ind_prods==i] = np.sum(r)/len(r)
    RD = np.fabs(ratings - avgRating)
    ################################################
    tprint('Computing EXT...')
    EXT = np.zeros((len(ratings),))
    EXT[np.logical_or(ratings == 5,ratings == 1)] = 1
    ################################################
    tprint('Computing DEV...')
    DEV = np.zeros((len(ratings),))
    normRD = RD/4
    DEV[normRD > DEV_BETA] = 1
    ################################################
    tprint('Computing ETF...')
    HRMat = sparse.csr_matrix((np.ones((len(ind_prods))), (ind_prods, ind_users)), shape=(len(uprod), len(uuser)))
    x, y = HRMat.nonzero()
    firstReviewDate = []
    for i in range(len(uprod)):
        ind = ind_prods == i
        d = int_dates[ind]
        firstReviewDate.append(d.min())
    F, ETF = np.zeros((len(ind_prods),)), np.zeros((len(ind_prods),))
    for i in range(len(x)):
        ind = np.logical_and(ind_prods == x[i], ind_users == y[i])
        d = int_dates[ind]
        deltaD = d.max() - firstReviewDate[x[i]]
        if deltaD <= ETF_DELTA:
            F[ind] = 1 - deltaD / ETF_DELTA
    ETF[F > ETF_BETA] = 1
    ################################################
    tprint('Computing ISR...')
    ISR = np.zeros((len(df),))
    for i in range(len(df)):
        ind = ind_users == i
        if np.sum(ind) == 1:
            ISR[ind] = 1

    df['f_RANK'] = pd.Series(Rank)
    df['f_RD'] = pd.Series(RD)
    df['f_EXT'] = pd.Series(EXT)
    df['f_DEV'] = pd.Series(DEV)
    df['f_ETF'] = pd.Series(ETF)
    df['f_ISR'] = pd.Series(ISR)

def process_review_text_features(df):
    from nltk import sent_tokenize, word_tokenize
    import string
    from textblob import TextBlob
    PP1_WORDS = set(['i', 'me', 'we', 'us', 'myself', 'ourselves', 'my', 'our', 'mine', 'ours'])
    PP2_WORDS = set(['you', 'your', 'yours', 'yourself', 'yourselves'])
    PUNCTUATION_SET = set(string.punctuation)
    ################################################
    tprint('Computing L...')
    L = df['word_list'].str.len()
    tprint('Computing PCW...')
    PCW = df['word_list'].apply(lambda word_list: len([1 for word in word_list if word.isupper()])) / L
    PCW.loc[(PCW > 1) | pd.isna(PCW)] = 0 # set divided by 0
    tprint('Computing PC...')
    PC = df['word_list'].apply(lambda word_list: len([1 for word in word_list for char in word if char.isupper()])) / df['word_list'].apply(lambda word_list: len([1 for word in word_list for char in word if char.isalpha()]))
    PC.loc[(PC > 1) | pd.isna(PC)] = 0 # set divided by 0
    tprint('Computing PP1...')
    _PP1 = df['word_list'].apply(lambda word_list: len([1 for word in word_list if word.lower() in PP1_WORDS]))
    _PP2 = df['word_list'].apply(lambda word_list: len([1 for word in word_list if word.lower() in PP2_WORDS]))
    PP1 = _PP1 / (_PP1 + _PP2)
    PP1.loc[(PP1 > 1) | pd.isna(PP1)] = 0  # set divided by 0
    tprint('Computing RES...')
    df_sentence_list = df['review_body'].apply(sent_tokenize)
    RES = df_sentence_list.apply(lambda sentence_list: len([1 for sentence in sentence_list if '!' in sentence])) / df_sentence_list.str.len()
    RES.loc[(RES > 1) | pd.isna(RES)] = 0  # set divided by 0
    tprint('Computing SW/OW...')
    df_subjectivity = df['word_list'].apply(lambda word_list: [TextBlob(word).subjectivity for word, tag in nltk.pos_tag(word_list) if tag.startswith('J') or tag.startswith('R') or tag.startswith('V')] )
    df_subjectivity_len = df_subjectivity.str.len()
    SW = df_subjectivity.apply(lambda subjectivity_list: len([1 for subjectivity in subjectivity_list if subjectivity == 1.0])) / df_subjectivity_len
    SW.loc[(SW > 1) | pd.isna(SW)] = 0  # set divided by 0
    OW = df_subjectivity.apply(lambda subjectivity_list: len([1 for subjectivity in subjectivity_list if subjectivity == 0.0])) / df_subjectivity_len
    OW.loc[(OW > 1) | pd.isna(OW)] = 0  # set divided by 0

    N = len(df)
    unigram_count = Counter(unigram for unigrams in df['word_list'] for unigram in unigrams)
    DL_u = df['word_list'].apply(lambda word_list: sum(-math.log(unigram_count[unigram]/N, 2) for unigram in word_list))
    df_bigrams = df_sentence_list.apply(lambda sentence_list: [bigram for sentence in sentence_list for bigram in nltk.bigrams(word for word in word_tokenize(sentence) if word not in PUNCTUATION_SET)])
    bigram_count = Counter(bigram for bigrams in df_bigrams for bigram in bigrams)
    DL_b = df_bigrams.apply(lambda bigram_list: sum(-math.log(bigram_count[bigram]/N, 2) for bigram in bigram_list))

    tprint('Assigning Features to Columns...')
    df['f_L'] = L
    df['f_PC'] = PC
    df['f_PCW'] = PCW
    df['f_PP1'] = PP1
    df['f_RES'] = RES
    df['f_SW'] = SW
    df['f_OW'] = OW
    df['f_DL_u'] = DL_u
    df['f_DL_b'] = DL_b

def process_gist_features(df, keyword_list):
    word_2_vec_model = Word2Vec.load(WORD_2_VEC_MODEL_NAME)
    keyword_vector_list = [word_2_vec_model.wv[keyword] for keyword in keyword_list]
    word_list_indexed = df['word_list'].apply(lambda word_list: [word for word in word_list if word_2_vec_model.wv.has_index_for(word)])
    for keyword, keyword_vector in zip(keyword_list, keyword_vector_list):
        tprint(f'Computing gist for {keyword}...')
        df[f"f_gist_{keyword}"] = word_list_indexed.apply(lambda word_list: np.min(word_2_vec_model.wv.distances(keyword_vector, word_list)) if len(word_list) else 1)
    tprint('Finished Gist Features')

def print_gist_features_stats(df, save=False):
    gist_columns = [col for col in df.columns if 'f_gist' in col]
    total_population = len(df)
    stats = []
    for i, col in enumerate(gist_columns):
        rounded = df[col].round(1)
        tprint(f"Gist {i} out of {len(gist_columns)}", col[7:])
        mean, max, min, median = round(df[col].mean(), 3), round(df[col].max(), 3), round(df[col].min(), 3), round(df[col].median(), 3)
        print("mean: ", mean, "max: ", max, "min: ", max, "median: ", median)
        distance_list, population_list = np.unique(rounded, return_counts=True)
        cum_genuine, cum_fraudulent, cum_ratio_list = 0, 0, []
        genuinity_list = [df.loc[rounded == distance, 'genuine'] for distance in distance_list]
        genuine_list = [sum(genuinity == 1) for genuinity in genuinity_list]
        fraudulent_list = [sum(genuinity == 0) for genuinity in genuinity_list]
        cumulative_genuine_list = [sum(genuine_list[:idx+1]) for idx in range(len((genuine_list)))]
        cumulative_fraudulent_list = [sum(fraudulent_list[:idx+1]) for idx in range(len((fraudulent_list)))]
        for distance, population, genuine, fraudulent, cum_genuine, cum_fraudulent in \
                zip(distance_list, population_list, genuine_list, fraudulent_list, cumulative_genuine_list, cumulative_fraudulent_list):
            cum_ratio_list.append(round(cum_genuine/cum_fraudulent, 2) if cum_fraudulent != 0 else np.nan)
            print(
                f"Distance={distance}; Population={population} ({round(population/total_population*100, 2)}%); "
                f"GTFR={round(genuine/fraudulent, 2) if fraudulent != 0 else 'ALL'} "
                f"(cumulative={cum_ratio_list[-1] if cum_ratio_list[-1] else 'ALL'})"
            )
        stats.append({
            'gist': col[7:],
            'distance_list': distance_list.tolist(),
            'population_list': population_list.tolist(),
            'genuine_list': genuine_list,
            'fraudulent_list': fraudulent_list,
            'mean': mean,
            'max': max,
            'min': min,
            'median': median
        })
        print("******************************")
    if save:
        with open(GIST_FEATURES_STATS_FILE_NAME, "w") as file:
            json.dump(stats, file)

def process_features(df, keyword_list):
    tprint('Processing Features...')
    updated = False
    TFIDF, W = None, None
    if not len([1 for col in df.columns if 'f_user' in col]) == 11:
        tprint('Processing User Features...')
        TFIDF, W = process_behaviour_features(df, type='user')
        updated = True
    if not len([1 for col in df.columns if 'f_product' in col]) == 10:
        tprint('Processing Product Features...')
        TFIDF, W = process_behaviour_features(df, type='product', TFIDF=TFIDF, W=W)
        updated = True
    del TFIDF, W
    if not all([col in df.columns for col in ['f_RANK', 'f_RD', 'f_EXT', 'f_DEV', 'f_ETF', 'f_ISR']]):
        tprint('Processing Review Behaviour Features...')
        process_review_behaviour_features(df)
        updated = True
    if not all([col in df.columns for col in ['f_PCW', 'f_PC', 'f_L', 'f_PP1', 'f_RES', 'f_SW', 'f_OW', 'f_DL_u', 'f_DL_b']]):
        tprint('Processing Review Behaviour Features...')
        process_review_text_features(df)
        updated = True
    if True or not all([f"f_gist_{keyword}" in df.columns for keyword in keyword_list]):
        tprint('Processing Gist Features...')
        process_gist_features(df, keyword_list)
        updated = True

    tprint('Finished all features')
    if updated:
        try:
            df.to_pickle(DF_FILE_NAME_WITH_FEATURES, compression={"method": "gzip", "compresslevel": 1})
            tprint("Successfully saved DataFrame")
        except:
            tprint("Failed to save DataFrame")

if __name__ == '__main__':
    df, df_polar, df_valid, genuine_ratio = load_dataset()
    exit()
    keyword_list = important_keywords(df_polar, genuine_ratio * 3)
    tprint("Keywords", keyword_list)
    process_gist_features(df, keyword_list)
    # print_gist_features_stats(df, save=True)
    # process_features(df)
    # model = train_word_2_vec_model()
    # save_word_2_vec_model(model)
    tprint('end')