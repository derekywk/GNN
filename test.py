import tensorflow_datasets as tfds
import pandas as pd
import spacy
import numpy as np
import gensim
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from datetime import datetime
from nltk.corpus import wordnet

DATASET = "Watches_v1_00"
# DATASET = "Shoes_v1_00"
DATASET_SIZE = 5000

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

def word_count(n=100):
    counter = Counter(' '.join(df['review_body']).lower().split())
    print("Most common 100 words", counter.most_common(n))

def using_spacy():
    nlp = spacy.load("en_core_web_sm")
    ents = nlp('\n'.join(df['review_body'])).ents
    ents_by_review = [nlp(review).ents for review in df['review_body'].str.decode('utf-8')]

def using_gensim_lda():
    docs = df['review_body'].str.decode('utf-8').map(lambda x: x.split()).to_numpy()
    # create dictionary of all words in all documents
    dictionary = gensim.corpora.Dictionary(docs)

    # filter extreme cases out of dictionary
    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # create BOW dictionary
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    # create LDA model using preferred hyperparameters
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                             num_topics=20,
                                             id2word=dictionary,
                                             passes=10,
                                             workers=4,
                                             random_state=21)
    # for each topic, print words occuring in that topic
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    path_to_model = "text_model"
    lda_model.save(path_to_model)

def keyword_extraction():
    import yake
    custom_kw_extractor = yake.KeywordExtractor(lan='en', n=2, dedupLim=0.9, top=20, features=None)
    keywords = custom_kw_extractor.extract_keywords('\n'.join(df['review_body']))

def pretrained_word_2_vec():
    model = api.load("glove-twitter-25")  # download the model and return as object ready for use
    # model = api.load("fasttext-wiki-news-subwords-300")

def train_word_2_vec_model() -> Word2Vec:
    tprint('Training Word2Vec Model...')
    return Word2Vec(df['words'])

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
    return col.str.lower().str.split(r"(?<=\w|-)[^\w-]+", regex=True).apply(
        lambda sentence: [
            lemmatizer.lemmatize(word, wordnet_tag(nltk_tag))
            for word, nltk_tag in nltk.pos_tag(sentence)
            if word not in words_to_remove and not word.isnumeric() and len(word) > 1
        ]
    )

def keyword_helpful_count(df, keyword, print=False):
    keyword_df = df.loc[df['words_set'].apply(lambda words: keyword in words)]
    genuine_count = keyword_df['genuine'].sum()
    fraud_count = sum(keyword_df['genuine'] == 0)
    if print == 'latex':
        print(fr"{keyword[2:]} & {len(keyword_df)} ({round(len(keyword_df) / len(df) * 100, 2)}\%) & {genuine_count} & {fraud_count} & {round(genuine_count / fraud_count, 2)}:1")
    elif print is True:
        print(f"{{{keyword}}}: Keyword Count: {len(keyword_df)} ({len(keyword_df)/len(df)}); Genuine Count: {genuine_count} ({genuine_count/len(df)}); Fraud Count: {fraud_count} ({fraud_count/len(df)}); ratio: {genuine_count/fraud_count}:1")
    else:
        return len(keyword_df), genuine_count, fraud_count, (genuine_count/fraud_count) if fraud_count else None

def common_words(df, topn=1000):
    counter = Counter(word for words in df['words'] for word in words)
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
        df = pd.read_csv(f"df_{DATASET}_size_{DATASET_SIZE}")
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
        df['words'] = preprocess(df["review_body"])

        tprint('Creating Words Sets...')
        df['words_set'] = df['words'].apply(set)

        df['genuine'] = None
        genuine = (df['helpful_votes'] / df['total_votes']) >= GENUINE_THRESHOLD
        fraud = (df['helpful_votes'] / df['total_votes']) <= FRAUDULENT_THRESHOLD
        df.loc[genuine, 'genuine'] = 1
        df.loc[fraud, 'genuine'] = 0

        try:
            df.to_csv(f"df_{DATASET}_size_{DATASET_SIZE}", index=False)
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

def build_user_features(df):
    df_by_user = df.groupby('customer_id', sort=False)
    for column in ['MNR', 'PR', 'NR', 'avgRD', 'WRD', 'BST', 'ERD', 'ETG', 'RL', 'ACS', 'MCS']:
        df[column] = np.nan
    MNR = {}
    for customer_id, user_df in df_by_user.__iter__():
        MNR[customer_id] = user_df.groupby('review_date')['review_id'].count().max()

    df['MNR'] = df['customer_id'].apply(lambda cusomter_id: MNR[customer_id])

if __name__ == '__main__':
    df, target_df, genuine_ratio = load_dataset()
    print(important_keywords(df, genuine_ratio * 3))
    model = train_word_2_vec_model()
    save_word_2_vec_model(model)
    tprint('end')