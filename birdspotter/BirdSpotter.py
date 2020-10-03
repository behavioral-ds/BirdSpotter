"""
birdspotter is a python package providing a toolkit to measures the social influence and botness of twitter users.
"""

import simplejson
import ijson
from tqdm import tqdm
import wget
import zipfile
import pandas as pd
import pickle as pk
import numpy as np
from birdspotter.utils import *
import collections
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
import dateutil
from birdspotter.user_influence import P, influence
import itertools
import multiprocessing as mp

class BirdSpotter:
    """Birdspotter measures the social influence and botness of twitter users.

    This class takes a twitter dump in (json or jsonl format) and extract metrics bot and influence metrics for the users. 
    The class will download word2vec embeddings if they are not specified. 
    It exposes processed data from the tweet dumps.
    
    Attributes:
        cascadeDataframe (:class:`pandas.DataFrame`): A dataframe of tweets ordered by cascades and time (the column casIndex denotes which cascade each tweet belongs to)
        featureDataframe (:class:`pandas.DataFrame`): A dataframe of users with their respective botness and influence scores.
        hashtagDataframe (:class:`pandas.DataFrame`): A dataframe of the text features for hashtags.

    """

    def __init__(self, path, tweetLimit = None, embeddings='download', quiet=False):
        """Birdspotter measures the social influence and botness of twitter users.
        
        Parameters
        ----------
        path : str
            The path to a tweet json or jsonl file containing the tweets for analysis.
        tweetLimit : int, optional
            A limit on the number of tweets to process if the tweet dump is too large, if None then all tweets are processed, by default None
        embeddings : collections.Mapping or str, optional
            A method for loading word2vec embeddings, which accepts are path to embeddings, a mapping object or a pickle object. Refer to set_word_embeddings for details. By default 'download'
        quiet : bool, optional
            Determines if debug statements will be printed or not, by default False
        """        
        self.word_embeddings = None
        self.quiet = quiet
        self.extractTweets(path,  tweetLimit = tweetLimit, embeddings=embeddings)

    def __pprint(self, message):
        """Print's statements to stdout if the quiet parameter is False 
        """
        if not self.quiet:
            print(message)

    def set_word_embeddings(self, embeddings='download', force_reload=True):
        """Sets the word2vec embeddings. The embeddings can be a path to a pickle or txt file, a mapping object or the string 'download' which will automatically download and use the FastText 'wiki-news-300d-1M.vec' if not available in the current path.
        
        Parameters
        ----------
        embeddings : collections.Mapping or str or None, optional
            A method for loading word2vec embeddings. A path to a embeddings pickle or txt file, a mapping object, or the string 'download', by default 'download'. If None, it does nothing.
        force_reload : bool, optional
            If the embeddings are already set, force_reload determines whether to update them, by default True
        """
        if not force_reload and self.word_embeddings is not None:
            return
        if embeddings is None:
            return
        elif isinstance(embeddings, str) and embeddings == 'download':
            if os.path.isfile('./wiki-news-300d-1M.vec'):
                # self.__pprint("Loading Fasttext wiki-news-300d-1M.vec Word2Vec Embeddings...")
                with open('./wiki-news-300d-1M.vec',"r") as f:
                    model = {}
                    if not self.quiet:
                        pbar = tqdm(total=1000000, desc="Loading Fasttext wiki-news-300d-1M.vec Word2Vec Embeddings")
                    for line in f:
                        splitLine = line.split()
                        word = splitLine[0]
                        embedding = np.array([float(val) for val in splitLine[1:]])
                        model[word] = embedding
                        if not self.quiet:
                            pbar.update(1)
                    if not self.quiet:
                        pbar.close()
                    self.word_embeddings = model
                # self.__pprint("Finished loading Word2Vec Embeddings")
            else:
                try:
                    self.__pprint("Downloading Fasttext embeddings")
                    filename = wget.download('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip')
                    self.__pprint('\n')
                    with zipfile.ZipFile(filename, 'r') as zip_ref:
                        zip_ref.extractall('./')
                    # self.__pprint("Loading downloaded Fasttext wiki-news-300d-1M.vec Word2Vec Embeddings...")
                    with open('./wiki-news-300d-1M.vec',"r") as f:
                        model = {}
                        if not self.quiet:
                            pbar = tqdm(total=1000000, desc="Loading downloaded Fasttext wiki-news-300d-1M.vec Word2Vec Embeddings")
                        for line in f:
                            splitLine = line.split()
                            word = splitLine[0]
                            embedding = np.array([float(val) for val in splitLine[1:]])
                            model[word] = embedding
                            if not self.quiet:
                                pbar.update(1)
                        if not self.quiet:
                            pbar.close()
                        self.word_embeddings = model
                    # self.__pprint("Finished loading Word2Vec Embeddings")
                except Exception as e:
                    print(e)
        elif isinstance(embeddings, str):
            embeddingsPath = embeddings
            _,fileextension = os.path.splitext(embeddingsPath)
            if fileextension == '.pickle':
                self.__pprint("Loading Word2Vec Embeddings...")
                with open(embeddingsPath,"rb") as f:
                    self.word_embeddings = pk.load(f)
                self.__pprint("Finished loading Word2Vec Embeddings")
            elif fileextension == '.txt':
                self.__pprint("Loading Word2Vec Embeddings...")
                with open(embeddingsPath,"r") as f:
                    model = {}
                    for line in f:
                        splitLine = line.split()
                        word = splitLine[0]
                        embedding = np.array([float(val) for val in splitLine[1:]])
                        model[word] = embedding
                    self.word_embeddings = model
                self.__pprint("Finished loading Word2Vec Embeddings")
        elif isinstance(embeddings, collections.Mapping):
            self.word_embeddings = embeddings
    
    def process_tweet(self, j):
        temp_user = {}
        temp_tweet = {}
        temp_text = (j['text'] if 'text' in j.keys() else j['full_text'])
        temp_content = {'status_text': temp_text, 'user_id' : j['user']['id_str']}
        temp_description = {'description':j['user']['description'], 'user_id' : j['user']['id_str']}
        temp_cascade = {}
        
        if 'retweeted_status' in j:
            temp_cascade['cascade_id'] = j['retweeted_status']['id_str']
            temp_cascade['original_created_at'] = j['retweeted_status']['created_at']
            temp_cascade['created_at'] = j['created_at']
            temp_cascade['retweeted'] = True
            original_tweet_processed = self.process_tweet(j['retweeted_status'])[0]
            # original_tweet_processed = None

        else:
            temp_cascade['cascade_id'] = j['id_str']
            temp_cascade['original_created_at'] = j['created_at']
            temp_cascade['created_at'] = j['created_at']
            temp_cascade['retweeted'] = False 

        temp_cascade['follower_count'] = j['user']['followers_count']
        temp_cascade['status_text'] = temp_text
        temp_cascade['screen_name'] = j['user']['screen_name']
        temp_cascade['hashtag_entities'] = [e['text'] for e in j['entities']['hashtags']]
        
        temp_user['screen_name'] = j['user']['screen_name']
        temp_user['url'] = j['user']['profile_image_url_https']
        temp_user['description'] = j['user']['description']
        temp_user['followers_count'] = j['user']['followers_count']    

        temp_cascade['user_id'] = j['user']['id_str']
        temp_user['user_id'] = j['user']['id_str']
        temp_tweet['user_id'] = j['user']['id_str']

        temp_user.update(getTextFeatures('name',j['user']['name']))
        temp_user.update(getTextFeatures('location',j['user']['location']))
        temp_user.update(getTextFeatures('description',j['user']['description']))
        for key in ['statuses_count', 'listed_count', 'friends_count', 'followers_count']:
            temp_user[key] = j['user'][key]
        temp_user['verified'] = 1 if j['user']['verified'] else 0
        temp_user['ff_ratio'] = (temp_user['followers_count'] + 1)/(temp_user['followers_count'] + temp_user['friends_count'] + 1)
        n = datetime.now()
        temp_user['years_on_twitter'] = (datetime(n.year, n.month, n.day) - datetime.strptime(j['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')).days/365
        temp_user['statuses_rate'] = (temp_user['statuses_count'] + 1)/(temp_user['years_on_twitter'] + .001)
        temp_user['tweets_to_followers'] = (temp_user['statuses_count'] + 1)/(temp_user['followers_count'] + 1)
        temp_user['favourites_count'] = j['user']['favourites_count']

        temp_tweet.update(getTextFeatures('status_text',temp_text))
        temp_tweet['n_tweets'] = 1 if 'retweeted_status' in j and ('quoted_status_is' in j) else 0
        temp_tweet['n_retweets'] = 1 if 'retweeted_status' in j else 0
        temp_tweet['n_quotes'] = 1 if 'quoted_status_id' in j else 0
        temp_tweet['n_timeofday'] = hourofweekday(j['created_at'])
        temp_tweet['retweet_count'] = j['retweet_count']
        temp_tweet['favorite_count'] = j['favorite_count']
        temp_tweet.update(getSource(j['source']))

        res = (temp_user, temp_tweet, temp_content,  temp_description,  temp_cascade)
        return ([res, original_tweet_processed] if 'retweeted_status' in j else [res])

    def extractTweets(self, filePath, tweetLimit = None, embeddings='download'):
        """Extracts tweets from a json or jsonl file and generates cascade, feature and hashtag dataframes as class attributes.
        
        Note that we use the file extension to determine how to handle the file.

        Parameters
        ----------
        filePath : str
            The path to a tweet json or jsonl file containing the tweets for analysis.
        tweetLimit : int, optional
            A limit on the number of tweets to process if the tweet dump is too large, if None then all tweets are processed, by default None
        embeddings : collections.Mapping or str or None, optional
            A method for loading word2vec embeddings. A path to a embeddings pickle or txt file, a mapping object, the string 'download', by default 'download'. If None, it does nothing.
        
        Returns
        -------
        DataFrame
            A dataframe of user's botness and influence scores (and other features).
        """        
        # self.__pprint("Starting Tweet Extraction")

        _,fileextension = os.path.splitext(filePath)
        raw_tweets = []

        def get_islice(iter):
            if tweetLimit != None:
                yield from itertools.islice(iter, tweetLimit)
            yield from iter
        
        with open(filePath, encoding="utf-8") as f:
            if fileextension == '.jsonl':
                raw_tweets = map(simplejson.loads, get_islice(f))
            elif fileextension == '.json':
                raw_tweets = get_islice(ijson.items(f, 'item'))
            else:
                    raise Exception('Not a valid tweet dump. Needs to be either jsonl or json, with the extension explicit.')
                # original_tweets = [j['retweeted_status'] for j in raw_tweets if 'retweeted_status' in j]
            # def run_in_parallel(func, seq):
            #     try:
            #         cpus = mp.cpu_count()
            #     except NotImplementedError:
            #         cpus = 4   # arbitrary default
            #     with mp.Pool(processes=cpus) as pool:
            #         return pool.apply(func, seq)
            
            user_list = []
            tweet_list = []
            w2v_content_list = []
            w2v_description_list = []
            cascade_list = []
            for temp_user, temp_tweet, temp_content,  temp_description,  temp_cascade in itertools.chain(*map(self.process_tweet, tqdm(raw_tweets, desc="Extracting raw tweets"))):
                user_list.append(temp_user)
                tweet_list.append(temp_tweet)
                w2v_content_list.append(temp_content)
                w2v_description_list.append(temp_description)
                cascade_list.append(temp_cascade)
            
        # We are assuming that user data doesn't change much and if it does, we take that 'latest' as our feature
        userDataframe = pd.DataFrame(user_list).fillna(0).set_index('user_id')
        userDataframe = userDataframe[~userDataframe.index.duplicated(keep='last')]
        
        tweetDataframe = pd.DataFrame(tweet_list).fillna(0).set_index('user_id')
        n_tweets = tweetDataframe['n_tweets'].groupby('user_id').sum()
        n_retweets = tweetDataframe['n_retweets'].groupby('user_id').sum()
        n_quoted = tweetDataframe['n_quotes'].groupby('user_id').sum()
        tweetDataframe = tweetDataframe.groupby('user_id').mean()
        tweetDataframe['n_retweets'] = n_retweets
        tweetDataframe['n_quotes'] = n_quoted
        tweetDataframe['n_tweets'] = n_tweets

        self.cascadeDataframe = pd.DataFrame(cascade_list).fillna(0)
        self.__reformatCascadeDataframe()

        contentDataframe = pd.DataFrame(w2v_content_list).set_index('user_id')
        descriptionDataframe = pd.DataFrame(w2v_description_list).set_index('user_id')
        descriptionDataframe = descriptionDataframe[~descriptionDataframe.index.duplicated(keep='last')]
        
        self.set_word_embeddings(embeddings, force_reload=False)
        self.featureDataframe = userDataframe.join(tweetDataframe)
        if self.word_embeddings is not None:
            w2vDataframe = self.__computeVectors(contentDataframe, descriptionDataframe)
            self.featureDataframe = self.featureDataframe.join(w2vDataframe)
        
        #Computes the features for all the hashtags. Is currently not protected from namespace errors.
        self.hashtagDataframe = self.__computeHashtagFeatures(self.cascadeDataframe)
        self.featureDataframe = self.featureDataframe.join(self.hashtagDataframe, rsuffix='_hashtag')
        self.featureDataframe = self.featureDataframe[~self.featureDataframe.index.duplicated()]
        return self.featureDataframe

    def getBotAnnotationTemplate(self, filename="annotationTemplate.csv"):
        """Writes a CSV with the list of users and a blank column "isbot" to be annotated.

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
        csv_data = self.cascadeDataframe.groupby(['screen_name', 'user_id']).apply(lambda d: '').reset_index(name='isbot')
        csv_data.to_csv(filename)
        return csv_data

    def __computeHashtagFeatures(self, cascadeDataframe):
        """Computes the hashtag tfidf features as a dataframe"""
        # hashtagSeries = contentdf['status_text'].str.findall(r'(?<!\w)#\w+').str.join(" ").str.replace("#","")
        tqdm.pandas(desc='Processing hashtag entities')
        hashtagSeries = cascadeDataframe.groupby('user_id').progress_apply(lambda g: " ".join(g['hashtag_entities'].apply(lambda x: " ".join(x))))
        userIndex = hashtagSeries.index
        crop = hashtagSeries.tolist()
        vectorizer = CountVectorizer(max_features=1000)
        X = vectorizer.fit_transform(crop)
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit(X).transform(X)
        column_names =  [x + '_hashtag' for x in vectorizer.get_feature_names()]
        hashtagDataframe = pd.DataFrame.sparse.from_spmatrix(tfidf, columns=column_names, index=userIndex)
        return hashtagDataframe

    def __computeVectors(self, contentdf, descriptiondf):
        """Computes the word2vec features as a dataframe"""
        ud = {}
        for index,row in contentdf.iterrows():
            vec = np.zeros(len(self.word_embeddings['a']))
            tol = 0
            for w in parse(row['status_text']):
                if w in self.word_embeddings:
                    vec = vec + np.array(self.word_embeddings[w])
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
            vec = np.zeros(len(self.word_embeddings['a']))
            tol = 0
            for w in parse(row['description']):
                if w in self.word_embeddings:
                    vec = vec + np.array(self.word_embeddings[w])
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
        """Loads the XGB booster model, from the saved XGB binary file
        
        Parameters
        ----------
        fname : str
            The path to the XGB binary file
        """        
        booster = xgb.Booster()
        booster.load_model(fname)
        self.booster = booster


    def loadPickledBooster(self, fname):
        """Loads the pickled booster model
        
        Parameters
        ----------
        fname : str
            The path to the pickled xgboost booster
        """        
        with open(fname, 'rb') as rf:
            return pk.load(rf)

    def trainClassifierModel(self, labelledData, targetColumnName='isbot', saveFileName=None, update=False, iterations=100, hyper_parameter_search=True):
        """Trains the bot detection classifier.

        Trains the bot detection classifier, using an XGB classifier. 
        Due to the way XGB works, the features used are the intersection, between the features from the tweet dumps and the features from the training set.

        Parameters
        ----------
        labelledData : str or pandas.DataFrame
            A path to the data with bot labels, as either csv or pickled dataframe, or a dataframe
        targetColumnName : str
            The name of the column, describing whether a user is a bot or not, by default 'isbot'
        saveFileName : str, optional
            The path of the file, to save the XGB model binary, which can be loaded with loadClassifierModel, by default None
        update : bool, optional
            Determines whether data will improve current classifier or restart training, by default False
        iterations : int, optional
            Determines the number of times the classifier training will iterate through data, by default 100
        hyper_parameter_search : bool, optional
            Determines if the hyper-parameters of the classifier should be search or if the default parameters will be used. The search may be time-consuming for large training datasets and doesn't work with update flag.

        """
        if isinstance(labelledData, str):
            _,fileextension = os.path.splitext(labelledData)
            if fileextension == '.csv':
                botrnot = pd.read_csv(labelledData, sep ="\t")
            elif fileextension == '.pickle':
                with open(labelledData,'rb') as f:
                    botrnot = pk.load(f)
        else:
            botrnot = labelledData
        if 'is_bot' in botrnot.columns:
            botTarget = botrnot['is_bot']
            del botrnot['is_bot']
        elif targetColumnName in botrnot.columns:
            botTarget = botrnot[targetColumnName]
            del botrnot[targetColumnName]
        else:
            raise Exception("The target column was not specified and cannot be found in the data. Please specify your target column accordingly.")
            
            
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
        no_features = len(botrnot.columns)
        tuned_parameters = {'n_estimators' : list(range(2, 300, 2)), 
                               'max_depth' : list(range(2, 100, 2)), 
                               'learning_rate' : [0.0001, 0.001, 0.01, 0.1],
                               'subsample':[0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
                              }
        if update:
            botrnot.reindex(columns=self.booster.feature_names)
            update = xgb.DMatrix(botrnot.values, botTarget.values, feature_names=botrnot.columns.values)
            self.booster.update(update, iterations)
        else:
            if hyper_parameter_search:
                clf = RandomizedSearchCV(estimator = XGBClassifier(), 
                                         param_distributions = tuned_parameters, 
                                         n_jobs = -1,
                                         n_iter=100,
                                         cv = 5)
                clf.fit(pd.DataFrame(data=botrnot.values,columns=botrnot.columns, index=botrnot.index), botTarget)
                self.booster = clf.best_estimator_.get_booster()
            else:
                train = xgb.DMatrix(botrnot.values, botTarget.values, feature_names=botrnot.columns.values)
                self.booster = xgb.train(params, train, iterations)
        if saveFileName is not None:
            self.booster.save_model(saveFileName)

    def getBotness(self):
        """Adds the botness of users to the feature dataframe. 
        
        It requires the tweets be extracted and the classifier be trained, otherwise exceptions are raised respectively.
        
        Returns
        -------
        DataFrame
            The current feature dataframe of users, with associated botness scores appended.
        
        Raises
        ------
        Exception
            Tweets haven't been extracted yet. Need to run extractTweets.
        """        
        if not hasattr(self, 'booster'):
            self.booster = self.loadPickledBooster(os.path.join(os.path.dirname(__file__), 'data', 'oversampled_booster.pickle'))
        if not hasattr(self, 'featureDataframe'):
            raise Exception("Tweets haven't been extracted yet")
        # Remove botness if it's already in the dataframe
        if 'botness' in self.featureDataframe.columns:
            del self.featureDataframe['botness']
        testdf = self.featureDataframe.reindex(columns=self.booster.feature_names)
        test = xgb.DMatrix(testdf.values, feature_names=self.booster.feature_names)
        bdf = pd.DataFrame()
        bdf['botness'] = self.booster.predict(test)
        bdf['user_id'] = testdf.index
        __botnessDataframe = bdf.set_index('user_id')
        self.featureDataframe = self.featureDataframe.join(__botnessDataframe) 
        self.featureDataframe = self.featureDataframe[~self.featureDataframe.index.duplicated()]
        return self.featureDataframe

    def __reformat_to_group(self, g):
            g = g.drop_duplicates(subset ="user_id", keep = 'last') 
            g = g.reset_index(drop=True)
            min_time = dateutil.parser.parse(g['original_created_at'][0])
            g['timestamp'] = pd.to_datetime(g['created_at']).values.astype('datetime64[s]')
            g['min_time'] = min_time
            g['min_time'] = g['min_time'].values.astype('datetime64[s]')
            g['diff'] = g['timestamp'].sub(g['min_time'], axis=0)
            g['time'] = g['diff'].dt.total_seconds()
            g['time'] = g['time'] - np.min(g['time'])
            g = g.sort_values(by=['time'])
            return g
    
    def __reformatCascadeDataframe(self):
        """ Reformats the cascade dataframe for influence estimation"""
        self.cascadeDataframe['magnitude'] = self.cascadeDataframe['follower_count']
        # self.__pprint('Reformatting cascades')
        tqdm.pandas(desc='Reformatting cascades')
        # Group the tweets by id
        if not self.quiet:
            self.cascadeDataframe = self.cascadeDataframe.groupby('cascade_id').progress_apply(self.__reformat_to_group)
        else:
            self.cascadeDataframe = self.cascadeDataframe.groupby('cascade_id').apply(self.__reformat_to_group)
        self.cascadeDataframe = self.cascadeDataframe.reset_index(drop=True)
        self.cascadeDataframe = self.cascadeDataframe[['magnitude','cascade_id', 'time', 'user_id', 'screen_name', 'status_text','created_at', 'hashtag_entities']]
        self.cascadeDataframe.sort_values(by=['cascade_id', 'time'])
        return self.cascadeDataframe


    def getInfluenceScores(self, params={'time_decay' : -0.000068, 'alpha' : None, 'beta' : 1.0}):
        """Adds a specified influence score to feature dataframe
        
        The specified influence will appear in the returned feature df, under the column 'influence (<alpha>,<time_decay>,<beta>)'.

        Parameters
        ----------
        time_decay : float, optional
            The time-decay r parameter described in the paper, by default -0.000068
        alpha : float, optional
             A float between 0 and 1, as described in the paper. If None DebateNight method is used, else spatial-decay method, by default None
        beta : float, optional
            A social strength hyper-parameter, by default 1.0
        
        Returns
        -------
        Dataframe
            The current feature dataframe of users, with associated botness scores.
        
        Raises
        ------
        Exception
            Tweets haven't been extracted yet. Need to run extractTweets.
        """        
        alpha, beta, time_decay = params['alpha'], params['beta'], params['time_decay']
        column_name = ("influence" if alpha == None and time_decay == -0.000068 and beta == 1.0 else 'influence ('+str(alpha)+','+str(time_decay)+','+str(beta)+')')
        if not hasattr(self, 'cascadeDataframe'):
            raise Exception("Tweets haven't been extracted yet")
        groups = self.cascadeDataframe.groupby('cascade_id')
        cascades = []
        # self.__pprint("Getting influence scores of users, with alpha of " + str(alpha) + ", with time decay of " + str(time_decay) + ", with beta of " + str(beta))
        if not self.quiet:
            pbar = tqdm(total=len(groups), desc="Getting influence scores of users, with alpha of " + str(alpha) + ", with time decay of " + str(time_decay) + ", with beta of " + str(beta))
        for i, g in groups:
            g = g.reset_index(drop=True)
            p = P(cascade=g, r=time_decay, beta=beta)
            inf, _ = influence(p, alpha)
            g[column_name] = pd.Series(inf)
            g['expected_parent'] = pd.Series(g['user_id'][list(np.argmax(p, axis=0))].values)
            cascades.append(g)
            if not self.quiet:
                pbar.update(1)
        if not self.quiet:
            pbar.close()
        self.cascadeDataframe = pd.concat(cascades)
        if column_name in self.featureDataframe.columns:
            del self.featureDataframe[column_name]
        tmp = self.cascadeDataframe.groupby(['user_id']).mean()
        tmp = tmp[[column_name]]
        self.featureDataframe = self.featureDataframe.join(tmp)
        self.featureDataframe = self.featureDataframe[~self.featureDataframe.index.duplicated()]
        return self.featureDataframe

    def getLabeledUsers(self, out=None):
        """Generates a standard dataframe of users with botness and DebateNight influence scores (and other features), and optionally outputs a csv.
        
        Parameters
        ----------
        out : str, optional
            A output path for a csv of the results, by default None
        
        Returns
        -------
        DataFrame
            A dataframe of the botness and influence scores (and other feautes) of each user
        
        Raises
        ------
        Exception
            Tweets haven't been extracted yet
        """
        if not hasattr(self, 'featureDataframe'):
            raise Exception("Tweets haven't been extracted yet")
        if not hasattr(self, 'booster'):
            self.booster = self.loadPickledBooster(os.path.join(os.path.dirname(__file__), 'data', 'oversampled_booster.pickle'))
        if 'botness' not in self.featureDataframe.columns:
            self.getBotness()
        if 'influence' not in self.featureDataframe.columns:
            self.getInfluenceScores()
        if 'cascade_membership' not in self.featureDataframe.columns:
            self.getCascadeMembership()
        if out is not None:
            self.featureDataframe.to_csv(out)
        return self.featureDataframe

    def getCascadeMembership(self):
        self.featureDataframe['cascade_membership'] = self.cascadeDataframe.groupby('user_id').apply(lambda x: list(x['cascade_id']))
        return self.featureDataframe

    def getCascadesDataFrame(self):
        """Adds botness column and standard influence to the cascade dataframe and returns the cascadeDataframe"""
        tmp = self.featureDataframe[['botness','influence']]
        tmp
        self.cascadeDataframe.drop([c for c in ['botness','influence'] if c in self.cascadeDataframe.columns], axis=1, inplace=True)
        self.cascadeDataframe = self.cascadeDataframe.join(tmp, on='user_id', lsuffix='l')
        self.cascadeDataframe.drop_duplicates(subset=['user_id','cascade_id'], inplace=True)
        return self.cascadeDataframe
