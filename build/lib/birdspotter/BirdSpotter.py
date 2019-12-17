"""
BirdSpotter is a module which provides an influence and bot detection toolkit for twitter.
"""

import simplejson
from tqdm import tqdm
import wget
import zipfile
import logging
import pandas as pd
import pickle as pk
import lzma
import numpy as np
from birdspotter.utils import *
# from utils import *
import traceback
import collections
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class BirdSpotter:
    """
    Influence and Bot Detection toolkit for twitter dumps.

    This module takes a twitter json dump and extract metrics bot and influence metrics for the users.
    It requires a a labelled dataset of bots to do bot detection. It exposes processed data from the tweet dumps.

    """

    def __init__(self):
        self.word2vecEmbeddings =  None

    def setWord2VecEmbeddings(self, embeddings=None, forceReload=True):
        """
        Sets the word2vec embeddings.

        Sets the word2vec embeddings if it hasn't alright been set, either through a python dict-like object or a path to a pickle or facebook text file.

        Parameters
        ----------
        embeddings : dict or str
            Either a python mapping object or a path to a pickle or facebook text file of the w2v embeddings
        forceReload : boolean
            If True then the modules embeddings are overridden, otherwise if they exist in the module they aren't reloaded

        """
        if not forceReload and self.word2vecEmbeddings is not None:
            return
        # if embeddings is None:
        #     # print("Loading Word2Vec Embeddings...")
        #     # with lzma.open("word2vec.xz","r") as f:
        #     #     self.word2vecEmbeddings = json.loads(f.read())
        #     print("Finished loading Word2Vec Embeddings")
        if embeddings is None:
            return
        elif isinstance(embeddings, str) and embeddings == 'download':
            if os.path.isfile('./wiki-news-300d-1M.vec'):
                print("Loading Facebook wiki-news-300d-1M.vec Word2Vec Embeddings...")
                with open('./wiki-news-300d-1M.vec',"r") as f:
                    model = {}
                    with tqdm(total=1000000) as pbar:
                        for line in f:
                            splitLine = line.split()
                            word = splitLine[0]
                            embedding = np.array([float(val) for val in splitLine[1:]])
                            model[word] = embedding
                            pbar.update(1)
                    self.word2vecEmbeddings = model
                print("Finished loading Word2Vec Embeddings")
            else:
                try:
                    print("Downloading Facebook embeddings")
                    filename = wget.download('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
                    print('\n')
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall('./')
                    print("Loading downloaded Facebook wiki-news-300d-1M.vec Word2Vec Embeddings...")
                    with open('./wiki-news-300d-1M.vec',"r") as f:
                        model = {}
                        with tqdm(total=1000000) as pbar:
                            for line in f:
                                splitLine = line.split()
                                word = splitLine[0]
                                embedding = np.array([float(val) for val in splitLine[1:]])
                                model[word] = embedding
                                pbar.update(1)
                        self.word2vecEmbeddings = model
                    print("Finished loading Word2Vec Embeddings")
                except Exception as e:
                    print(e)
        elif isinstance(embeddings, str):
            embeddingsPath = embeddings
            _,fileextension = os.path.splitext(embeddingsPath)
            if fileextension == '.pickle':
                print("Loading Word2Vec Embeddings...")
                with open(embeddingsPath,"rb") as f:
                    self.word2vecEmbeddings = pk.load(f)
                print("Finished loading Word2Vec Embeddings")
            elif fileextension == '.txt':
                print("Loading Word2Vec Embeddings...")
                with open(embeddingsPath,"r") as f:
                    model = {}
                    for line in f:
                        splitLine = line.split()
                        word = splitLine[0]
                        embedding = np.array([float(val) for val in splitLine[1:]])
                        model[word] = embedding
                    self.word2vecEmbeddings = model
                print("Finished loading Word2Vec Embeddings")
        elif isinstance(embeddings, collections.Mapping):
            self.word2vecEmbeddings = embeddings
    
    def extractTweets(self, filePath, tweetLimit = None, embeddings='download'):
        """
        Extracts tweets from a json dump into a pandas dataframe.

        Parameters
        ----------
        filePath : str
            The path to the json twitter dump, to be loaded.
        tweetLimit : int
            Sets a limit on the number of tweets read.
        embeddings : dict or str
            Either a python mapping object or a path to a pickle or facebook text file of the w2v embeddings

        Returns
        -------
        Dataframe
            A dataframe of the features for each user

        """
        # Appending DataFrames line by line is inefficient, because it generates a
        # new dataframe each time. It better to get the entire list and them concat.
        user_list = []
        tweet_list = []
        w2v_content_list = []
        w2v_description_list = []
        cascade_list = []
        print("Starting Tweet Extraction")
        with open(filePath, encoding="utf-8") as f:
            for i, line in enumerate(f,1):
                if tweetLimit is not None and tweetLimit < i:
                    break
                j = simplejson.loads(line)
                try:
                    temp_user = {}
                    temp_tweet = {}
                    temp_content = {'status_text':j['text'], 'user_id' : j['user']['id']}
                    temp_description = {'description':j['user']['description'], 'user_id' : j['user']['id']}
                    temp_cascade = {}
                    
                    if 'retweeted_status' in j:
                        temp_cascade['cascade_id'] = j['retweeted_status']['id']
                        temp_cascade['original_created_at'] = j['retweeted_status']['created_at']
                        temp_cascade['created_at'] = j['created_at']
                        temp_cascade['retweeted'] = True
                    else:
                        temp_cascade['cascade_id'] = j['id']
                        temp_cascade['original_created_at'] = j['created_at']
                        temp_cascade['created_at'] = j['created_at']
                        temp_cascade['retweeted'] = False    
                    temp_cascade['follower_count'] = j['user']['followers_count']
                    temp_cascade['status_text'] = j['text']
                    temp_cascade['screen_name'] = j['user']['screen_name']
                    
                    temp_cascade['user_id'] = j['user']['id']
                    temp_user['user_id'] = j['user']['id']
                    temp_tweet['user_id'] = j['user']['id']

                    temp_user.update(getTextFeatures('name',j['user']['name']))
                    temp_user.update(getTextFeatures('location',j['user']['location']))
                    temp_user.update(getTextFeatures('description',j['user']['description']))
                    for key in ['statuses_count', 'listed_count', 'friends_count', 'followers_count']:
                        temp_user[key] = j['user'][key]
                    temp_user['verified'] = 1 if j['user']['verified'] else 0
                    temp_user['ff_ratio'] = (temp_user['followers_count'] + 1)/(temp_user['followers_count'] + temp_user['friends_count'] + 1)
                    temp_user['years_on_twitter'] = (datetime.now() - datetime.strptime(j['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')).days/365
                    temp_user['statuses_rate'] = (temp_user['statuses_count'] + 1)/(temp_user['years_on_twitter'] + .001)
                    temp_user['tweets_to_followers'] = (temp_user['statuses_count'] + 1)/(temp_user['followers_count'] + 1)
                    temp_user['retweet_count'] = j['retweet_count']
                    temp_user['favorite_count'] = j['favorite_count']
                    temp_user['favourites_count'] = j['user']['favourites_count']

                    temp_tweet.update(getTextFeatures('status_text',j['text']))
                    temp_tweet['n_tweets'] = 1 if 'retweeted_status' in j and ('quoted_status_is' in j) else 0
                    temp_tweet['n_retweets'] = 1 if 'retweeted_status' in j else 0
                    temp_tweet['n_quotes'] = 1 if 'quoted_status_id' in j else 0
                    temp_tweet['n_timeofday'] = hourofweekday(j['created_at'])
                    temp_tweet.update(getSource(j['source']))

                    user_list.append(temp_user)
                    tweet_list.append(temp_tweet)
                    w2v_content_list.append(temp_content)
                    w2v_description_list.append(temp_description)
                    cascade_list.append(temp_cascade)
                except Exception as err:
                    traceback.print_tb(err.__traceback__)
        # We are assuming that user data doesn't change much and if it does, we take that 'latest' as our feature
        userDataframe = pd.DataFrame(user_list).fillna(0).set_index('user_id')
        userDataframe = userDataframe[~userDataframe.index.duplicated(keep='last')]
        
        tweetDataframe = pd.DataFrame(tweet_list).fillna(0).set_index('user_id')
        n_retweets = tweetDataframe['n_retweets'].groupby('user_id').sum()
        n_quoted = tweetDataframe['n_quotes'].groupby('user_id').sum()
        tweetDataframe = tweetDataframe.groupby('user_id').mean()
        tweetDataframe['n_retweets'] = n_retweets
        tweetDataframe['n_quotes'] = n_quoted

        self.cascadeDataframe = pd.DataFrame(cascade_list).fillna(0)

        contentDataframe = pd.DataFrame(w2v_content_list).set_index('user_id')
        descriptionDataframe = pd.DataFrame(w2v_description_list).set_index('user_id')
        descriptionDataframe = descriptionDataframe[~descriptionDataframe.index.duplicated(keep='last')]
        
        self.setWord2VecEmbeddings(embeddings, forceReload=False)
        self.featureDataframe = userDataframe.join(tweetDataframe)
        if self.word2vecEmbeddings is not None:
            w2vDataframe = self.__computeVectors(contentDataframe, descriptionDataframe)
            self.featureDataframe = self.featureDataframe.join(w2vDataframe)
        
        #Computes the features for all the hashtags. Is currently not protected from namespace errors.
        self.hashtagdf = self.__computeHashtagFeatures(contentDataframe)
        self.featureDataframe = self.featureDataframe.join(self.hashtagdf)
        return self.featureDataframe

    def getBotAnnotationTemplate(filename="annotationTemplate.csv"):
        """
        Writes a CSV with the list of users and a blank column "isbot" to be annotated.

        A helper function which outputs a CSV to be annotated by a human. The output is a list of users with the blank "isbot" column.

        Parameters
        ----------
        filename : str
            The name of the file to write the CSV

        Returns
        -------
        Dataframe
            A dataframe of the users, with their screen names and a blank "is_bot" column.

        """
        csv_data = (self.cascadeDataframe.groupby(['screen_name', 'user_id']).apply(lambda d: '').reset_index(name='isbot'))
        csv_data.to_csv(filename)
        return csv_data

    def __computeHashtagFeatures(self, contentdf):
        """Computes the hashtag tfidf features as a dataframe"""
        hashtagSeries = contentdf['status_text'].str.findall(r'(?<!\w)#\w+').str.join(" ").str.replace("#","")
        userIndex = hashtagSeries.index
        crop = hashtagSeries.tolist()
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(crop)
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(X)
        column_names = vectorizer.get_feature_names()
        hashtagdf = pd.DataFrame(tfidf.toarray(), columns=column_names, index=userIndex)
        return hashtagdf

    def __computeVectors(self, contentdf, descriptiondf):
        """Computes the word2vec features as a dataframe"""
        ud = {}
        for index,row in contentdf.iterrows():
            vec = np.zeros(300)
            tol = 0
            for w in parse(row['status_text']):
                if w in self.word2vecEmbeddings:
                    vec = vec + np.array(self.word2vecEmbeddings[w])
                    tol += 1
            if tol != 0 and not np.isnan(tol):
                vec = vec/tol
            if index in ud:
                ud[index].append(vec)
            else:
                ud[index] = [vec]
        for k,v in ud.items():
            ud[k] = np.array(v).mean(axis=0)
        conw2v = pd.DataFrame(ud)
        conw2v = conw2v.T
        conw2v.index.name = 'user_id'
        conw2v.columns = ["con_w2v_" + str(i) for i in conw2v.columns]
        
        ud = {}
        for index,row in descriptiondf.iterrows():
            vec = np.zeros(300)
            tol = 0
            for w in parse(row['description']):
                if w in self.word2vecEmbeddings:
                    vec = vec + np.array(self.word2vecEmbeddings[w])
                    tol += 1
            if tol != 0 and not np.isnan(tol):
                vec = vec/tol
            ud[index] = [vec]
        for k,v in ud.items():
            ud[k] = np.array(v).mean(axis=0)
        desw2v = pd.DataFrame(ud)
        desw2v = desw2v.T
        desw2v.index.name = 'user_id'
        desw2v.columns = ["des_w2v_" + str(i) for i in desw2v.columns]

        return conw2v.join(desw2v)
    
    def loadClassifierModel(self, fname):
        """Loads the XGB model, from the saved XGB binary file"""
        clf = XGBClassifier(
        learning_rate =0.1,
        n_estimators=80,
        max_depth=5, #16
        subsample=0.6,
        colsample_bytree=1,
        objective= 'binary:logistic',
        n_jobs=10,
        silent=True,
        seed =27
        )
        clf.load_model(fname)
        self.clf = clf

    def trainClassifierModel(self, labelledDataPath, targetColumnName='isbot', saveFileName=None):
        """
        Trains the bot detection classifier.

        Trains the bot detection classifier, using an XGB classifier. 
        Due to the way XGB works, the features used are the intersection, between the features from the tweet dumps and the features from the training set.

        Parameters
        ----------
        labelledDataPath : str
            A path to the data with bot labels, as either csv or pickled dataframe/
        targetColumnName : str
            The name of the column, describing whether a user is a bot or not
        saveFileName : str
            The name of the file, to save the XGB model binary. It can be loaded with loadClassifierModel.

        """
        params = {
        'learning_rate' :0.1,
        'n_estimators':80,
        'max_depth':5, #16
        'subsample':0.6,
        'colsample_bytree':1,
        'objective': 'binary:logistic',
        'n_jobs':10,
        'silent':True,
        'seed' :27
        }

        _,fileextension = os.path.splitext(labelledDataPath)
        if fileextension == '.csv':
            botrnot = pd.read_csv(labelledDataPath, sep ="\t")
        elif fileextension == '.pickle':
            with open(labelledDataPath,'rb') as f:
                botrnot = pk.load(f)
        if 'is_bot' in botrnot.columns:
            botTarget = botrnot['is_bot']
        elif targetColumnName in botrnot.columns:
            botTarget = botrnot[targetColumnName]
        else:
            raise Exception("The target column was not specified and cannot be found in the data. Please specify your target column accordingly.")
        self.columnNameIntersection = list(set(self.featureDataframe.columns.values).intersection(set(botrnot.columns.values)))
        botrnot = botrnot[self.columnNameIntersection]
        train = xgb.DMatrix(botrnot.values, botTarget.values, feature_names=botrnot.columns.values)
        self.clf = xgb.train(params, train, 80)
        if saveFileName is not None:
            self.clf.save_model(saveFileName)

    def getBotness(self):
        """ Returns a dataframe of the botness of each user in the tweet dump. Exposes botnessDataframe through module."""
        if self.clf is None:
            raise Exception("The classifier has not been loaded yet")
        if self.featureDataframe is None:
            raise Exception("Tweets haven't been extracted yet")
        if self.columnNameIntersection is None:
            with open(os.path.join(os.path.dirname(__file__), 'data', 'standard_bot_features'), 'r') as f:
                standard_bot_features = set(f.read().splitlines())
                self.columnNameIntersection = list(set(self.featureDataframe.columns.values).intersection(standard_bot_features))
        testdf = self.featureDataframe[self.columnNameIntersection]
        test = xgb.DMatrix(testdf.values, feature_names=self.columnNameIntersection)
        bdf = pd.DataFrame()
        bdf['botness'] = self.clf.predict(test)
        bdf['user_id'] = testdf.index
        self.botnessDataframe = bdf.set_index('user_id')
        return self.botnessDataframe

    def composeData(self):
        """Adds botness column to the cascade dataframe. Exposes composedDataframe through module."""
        new = None
        if self.botnessDataframe is not None and self.cascadeDataframe is not None:
            new = self.botnessDataframe.loc[self.cascadeDataframe['user_id']].reset_index()['botness']
            new = pd.concat([self.cascadeDataframe,new], ignore_index=True, axis=1)
            new.columns = list(self.cascadeDataframe.columns.values) + ['botness']
            self.composedDataframe = new
        return new
