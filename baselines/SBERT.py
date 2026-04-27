# -*- coding: UTF-8 -*-
# @Project -> File: run_offline model.py -> lda2vec
# @Time: 3/8/23 00:42 
# @Author: Yu Yongsheng
# @Description: SBERT embedding

import numpy as np
import pandas as pd
import os
import datetime
import torch
import gensim
# --------------------------------load_tweet_data------------------------------------
project_path = os.getcwd()

def SBERT(dataset_name, i):
    print('dataset: ', dataset_name)
    load_path = project_path + '/data/raw dataset/'
    save_path = project_path + f'/data/{dataset_name}_offline_embeddings/block_' + str(i)
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
        df = pd.DataFrame(data=df, columns=['tweet_id', 'user_id', 'user_mentions','text', 'created_at',
                                                    'place_name', 'place_id', 'event_id', 'words',
                                                    'filtered_words', 'entities', 'sampled_words'])
        df = df[df['filtered_words'].apply(lambda x: len(x) > 0)]
        # sort date by time
        df = df.sort_values(by='created_at').reset_index(drop=True)
        df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')
        df['date'] = df['created_at'].dt.date  # Use the dt accessor to extract date
        init_day = df.loc[0, 'date']
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

    # SBERT embedding
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_words = df.filtered_words.apply(lambda x: ' '.join(x)).tolist()
    # Pass inputs through Bert model
    sbert_embeddings = sbert_model.encode(doc_words, convert_to_numpy=True)
    print('SBERT vectors shape: {}.'.format(sbert_embeddings.shape))
    return sbert_embeddings