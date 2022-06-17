import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import spacy
import numpy as np
import gensim
from collections import Counter
from gensim import corpora, models
ds, info = tfds.load('amazon_us_reviews/Watches_v1_00', split='train', with_info=True)
# ds, info = tfds.load('amazon_us_reviews/Shoes_v1_00', split='train', with_info=True)
df = pd.DataFrame(tfds.as_dataframe(ds, info))
df = df.rename({col: col.replace('data/', '') for col in df.columns}, axis=1)
df['review_body'] = df['review_body'].str.decode('utf-8')
genuine = (df['helpful_votes'] / df['total_votes']) >= 0.7
fraud = (df['helpful_votes'] / df['total_votes']) <= 0.3
target_df = df.loc[genuine|fraud]
print("Size of Dataset", df.shape)
print(f"Genuine Count: {sum(genuine)} ({sum(genuine)/len(target_df)}); Fraud Count: {sum(fraud)} ({sum(fraud)/len(target_df)}); ratio: {sum(genuine)/sum(fraud)}:1")

# WATCH DATASET
# Size of Dataset (960872, 15)
# Genuine Count: 247142 (0.2572059545912463); Fraud Count: 73549 (0.07654401418711337); ratio: 3.360236033120777:1

def keyword_helpful_count(keyword, latex=True):
    keyword_df = target_df.loc[target_df['review_body'].str.contains(keyword, case=False)]
    genuine = (keyword_df['helpful_votes'] / keyword_df['total_votes']) >= 0.7
    fraud = (keyword_df['helpful_votes'] / keyword_df['total_votes']) <= 0.3
    if latex:
        print(fr"{keyword[2:]} & {len(keyword_df)} ({round(len(keyword_df)/len(target_df)*100, 2)}\%) & {sum(genuine)} & {sum(fraud)} & {round(sum(genuine)/sum(fraud), 2)}:1")
    else:
        print(f"{{{keyword}}}: Keyword Count: {len(keyword_df)} ({len(keyword_df)/len(target_df)}); Genuine Count: {sum(genuine)} ({sum(genuine)/len(target_df)}); Fraud Count: {sum(fraud)} ({sum(fraud)/len(target_df)}); ratio: {sum(genuine)/sum(fraud)}:1")
    return sum(genuine), sum(genuine)/len(target_df), sum(fraud), sum(fraud)/len(target_df)

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

def word_2_vec():
    import gensim.downloader as api
    info = api.info()  # show info about available models/datasets
    model = api.load("glove-twitter-25")  # download the model and return as object ready for use
    # model = api.load("fasttext-wiki-news-subwords-300")

if __name__ == '__main__':
    using_gensim_lda()