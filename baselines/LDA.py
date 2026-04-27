# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> lda2vec
# @Time: 3/8/23 00:42 
# @Author: Yu Yongsheng
# @Description: gensim 调包实现 lda2vec embedding

import numpy as np
import pandas as pd
import os
import datetime
import torch
import gensim
# --------------------------------load_tweet_data------------------------------------
project_path = os.getcwd()
# -----------------------------------------lda2vec embeddings------------------------------------------------
from gensim.corpora import Dictionary
from gensim.models import LdaModel

train_vec = project_path + '/baselines/lda_model'
def train_lda(tokenized_docs, num_topics=256):
    """
    tokenized_docs: List[List[str]]，例如 df['filtered_words'].tolist()
    num_topics: 主题数，同时也是最终 embedding 的维度
    """
    # 1. construct dictionary
    '''dictionary is word->ID mapping list, Gensim.Dictionary会给each word分配一个unique id，存储成一个word list.'''
    dictionary = Dictionary(tokenized_docs)
    # optional, filter extra low-/high-frequency words, avoiding noise.
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    # 2. construct bow corpus
    '''doc2bow is 词袋模型 bag-of-words, 将each message (token list) mapping to 稀疏词频向量 with help of dictionary, [(help->0, 1 frequency), (3,2)]'''
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    # 3. training LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        iterations=100,
        random_state=42,
    )
    # 4. save model
    lda_model.save(train_vec + f'_{num_topics}_topics.model')
    dictionary.save(train_vec + f'_{num_topics}_dict.pkl')

    return lda_model, dictionary

def LDA(dataset_name, i):
    print('dataset: ', dataset_name)
    load_path = project_path + '/data/raw dataset/'
    save_path = project_path + f'/data/{dataset_name}_offline_embeddings/block_{i}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if dataset_name in ['Twitter', 'Twitter_4days']:
        # load data (68841 tweets, multiclasses filtered)
        p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
        p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
        # allow_pickle: 可选，布尔值，允许使用 Python pickles 保存对象数组，Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化。
        np_part1 = np.load(p_part1, allow_pickle=True)  # (35000, 16)
        np_part2 = np.load(p_part2, allow_pickle=True)  # (33841, 16)

        np_tweets = np.concatenate((np_part1, np_part2), axis=0)  # (68841, 16)
        print('Data loaded.')

        df = pd.DataFrame(data=np_tweets, columns=['event_id', 'tweet_id', 'text', 'user_id', 'created_at', 'user_loc',
                                                   'place_type', 'place_full_name', 'place_country_code', 'hashtags',
                                                   'user_mentions', 'image_urls', 'entities', 'words', 'filtered_words',
                                                   'sampled_words'])
        print('Data converted to dataframe.')
        # sort date by time
        df = df.sort_values(by='created_at').reset_index(drop=True)

        # append date
        df['date'] = [d.date() for d in df['created_at']]
        # 因为graph太大，爆了内存，所以取4天的twitter data做demo，后面用nci server
        init_day = df.loc[0, 'date']
        if dataset_name == 'Twitter_4days':
            df = df[(df['date'] >= init_day + datetime.timedelta(days=i)) & (
                    df['date'] <= init_day + datetime.timedelta(days=int(i+3)))].reset_index(drop=True)  # (11971, 18)
        else:  # 2days
            df = df[(df['date'] >= init_day + datetime.timedelta(days=i)) & (
                    df['date'] <= init_day + datetime.timedelta(days=int(i + 1)))].reset_index(drop=True)  # (11971, 18)
        print(df.shape)
        print(df.event_id.nunique())
        print(df.user_id.nunique())
    elif dataset_name in ['CrisisLexT', 'CrisisLexT_30M', 'Kawarith']:
        if dataset_name == 'CrisisLexT_30M':
            df = np.load(os.path.join(load_path, 'CrisisLexT.npy'.format(dataset_name)), allow_pickle=True)  # (5802, 302)
        else:
            df = np.load(os.path.join(load_path, '{}.npy'.format(dataset_name)), allow_pickle=True)  # (5802, 302)
        df = pd.DataFrame(data=df, columns=['tweet_id', 'user_id', 'user_mentions', 'text', 'created_at',
                                                    'place_name', 'place_id', 'event_id', 'words',
                                                    'filtered_words', 'entities', 'sampled_words'])
        df = df[df['filtered_words'].apply(lambda x: len(x) > 0)]
        if dataset_name == 'CrisisLexT':
            # Get the first two unique classes in 'event_id'
            first_two_classes = [0, 4]
            # Filter the DataFrame to include only messages from these two classes
            df = df[df['event_id'].isin(first_two_classes)].reset_index(drop=True)
        elif dataset_name == 'CrisisLexT_30M':
            # Get the first two unique classes in 'event_id'
            first_two_classes = [0, 4, 6]
            # Filter the DataFrame to include only messages from these two classes
            df = df[df['event_id'].isin(first_two_classes)].reset_index(drop=True)
        print(df['event_id'].value_counts())
        print(df.shape)  # (4762, 18)
        print(df.event_id.nunique())  # 57
        print(df.user_id.nunique())  # 4355
    elif dataset_name == 'French':
        df = np.load(os.path.join(load_path, 'All_French.npy'), allow_pickle=True)  # (5802, 302)
        df = pd.DataFrame(data=df, columns=["tweet_id", "user_id", "text", "time", "event_id", "user_mentions",
                                            "hashtags", "urls", "words", "created_at", "filtered_words", "entities", "sampled_words"])
        df = df[df['filtered_words'].apply(lambda x: len(x) > 0)] # del null record in filtered_words
        # sort event_id
        df = df.sort_values(by='created_at').reset_index(drop=True)
        df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')
        df['date'] = df['created_at'].dt.date  # Use the dt accessor to extract date
        init_day = df.loc[0, 'date']
        df = df[(df['date'] >= init_day + datetime.timedelta(days=i)) & (
                    df['date'] <= init_day + datetime.timedelta(days=i+2))].reset_index(drop=True)  # (11971, 18)
        print(df['event_id'].value_counts())
        print(df.shape)  # (4762, 18)
        print(df.event_id.nunique())  # 57
        print(df.user_id.nunique())  # 4355

    # ---------training LDA---------------------------------
    # use whole df training, LDA is unsupervised
    num_topics = 256
    tokenized_docs = df['filtered_words'].tolist()
    lda_model, dictionary = train_lda(tokenized_docs, num_topics=num_topics)
    # generate topic distribution of each message
    corpus_all = [dictionary.doc2bow(text) for text in tokenized_docs]

    doc_topic_vectors = []
    for bow in corpus_all:
        # 主题分布: get_document_topics return (topic_id, probability) list，计算each doc属于each topic的概率p(topic=k|d) with Gibbs sampling.
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        dense_vec = np.zeros(num_topics, dtype=np.float32)
        for topic_id, prob in topic_dist:
            dense_vec[topic_id] = prob  # 创建一个256-topic lda embedding, each dimension corresponds to a topic, record each message's prob.
        doc_topic_vectors.append(dense_vec)

    lda_embeddings = np.stack(doc_topic_vectors, axis=0)
    print(lda_embeddings.shape)
    np.save(save_path + '/lda_embeddings.npy', lda_embeddings)
    return lda_embeddings