from datetime import datetime
import re
import string

def parse(x):
    if type(x) != str:
        return []
    x =  re.sub("https?:","",x)
    x =  x.lower()
    words =  re.findall("[a-z]+",x)
    digs = re.findall("\\d",x)
    punc =  re.findall(r"""[!|"|#|$|%|&|\'|(|)|\*|\+|,|-|.|\/|:|;|<|=|>|\?|@|\[|\\|\]|^|_|`|{|||}|~]""",x)
    return words + digs + punc

def hourofweekday(datestring):
    d = datetime.strptime(datestring, '%a %b %d %H:%M:%S +0000 %Y')
    return d.weekday()*24 + d.hour + d.minute/60 + d.second/360

def grep(sourcestring, pattern):
    return 1 if pattern.lower() in sourcestring.lower() else 0

def getSource(sourcestring):
    sources = [('google','google'),('ifttt','IFTTT'),('facebook','facebook'),('ipad','for iPad'),('lite','twitter lite'),('hootsuite','hootsuite'),('android','android'),('webclient','web client'),('iphone','iphone')]
    return { x:grep(sourcestring,y) for x,y in sources}

def getURLs(string): 
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    return url 

    #No idea what lowersp or capsp is, but assume it means percentage
def getTextFeatures(key, text):
    res = {}
    if text is not None:
        res[key+'_n_chars'] = len(text)
        res[key+'_n_commas'] = text.count(",")
        res[key+'_n_digits'] = sum([x.isdigit() for x in list(text)])
        res[key+'_n_exclaims'] = sum([x=='!' for x in list(text)])
        res[key+'_n_extraspaces'] = sum([x==' ' for x in list(text)])
        res[key+'_n_hashtags'] = sum([x=='#' for x in list(text)])
        res[key+'_n_lowers'] = sum([x.islower() for x in list(text)])
        res[key+'_n_mentions'] = sum([x=='@' for x in list(text)])
        res[key+'_n_periods'] = sum([x=='.' for x in list(text)])
        if key != 'name':
            res[key+'_n_urls'] = len(getURLs(text))
        res[key+'_n_words'] = len(re.sub("[^\w]", " ",  text).split())
        res[key+'_n_caps'] = sum([x.isupper() for x in list(text)])
        res[key+'_n_nonasciis'] = sum([ord(x) < 128 for x in list(text)])
        res[key+'_n_puncts'] = sum([x in string.punctuation for x in list(text)])
        res[key+'_n_charsperword'] = (len(text)+1)/(res[key+'_n_words']+1)
        res[key+'_n_lowersp'] = (res[key+'_n_lowers']+1)/(res[key+'_n_chars'] + 1)
        res[key+'_n_capsp'] = (res[key+'_n_caps'] + 1)/(res[key+'_n_chars'] + 1)
    return res