import io
import random
import re
import string
import warnings
import tkinter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia as wk
import warnings
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr
import nltk
nltk.download()
from nltk.stem import WordNetLemmatizer

# for downloading package files can be commented after First run
nltk.download('popular', quiet=True)
nltk.download('nps_chat', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

posts = nltk.corpus.nps_chat.xml_posts()[:10000]


# To Recognise input type as QUES.
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features


featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# colour palet
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


def prYellow(skk): print("\033[93m {}\033[00m".format(skk))


def prLightPurple(skk): print("\033[94m {}\033[00m".format(skk))


def prPurple(skk): print("\033[95m {}\033[00m".format(skk))


def prCyan(skk): print("\033[96m {}\033[00m".format(skk))


def prLightGray(skk): print("\033[97m {}\033[00m".format(skk))


def prBlack(skk): print("\033[98m {}\033[00m".format(skk))


# Reading in the input_corpus
with open('HR.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# TOkenisation
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response and processing
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if "tell me about" in user_response:
        print("Checking Wikipedia")
        if user_response:
            robo_response = wikipedia_data(user_response)
            return robo_response
    elif (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response  # wikipedia search


def wikipedia_data(input):
    reg_ex = re.search('tell me about (.*)', input)
    try:
        if reg_ex:
            topic = reg_ex.group(1)
            wiki = wk.summary(topic, sentences=3)
            return wiki
    except Exception as e:
        print("No content has been found")


# Recording voice input using microphone
file = "file.mp3"
flag = True
fst = "My name is Chatter. I will answer your queries about Science. If you want to exit, say Bye"
tts = gTTS(text=fst, lang="en", tld="com")
tts.save("file.mp3")
os.system("mpg123 file.mp3")
r = sr.Recognizer()
prYellow(fst)

# Taking voice input and processing
while (flag == True):
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        user_response = format(r.recognize_google(audio))
        print("\033[91m {}\033[00m".format("YOU SAID : " + user_response))
    except sr.UnknownValueError:
        prYellow("Oops! Didn't catch that")
        pass

    user_response = input()
    user_response = user_response.lower()
    clas = classifier.classify(dialogue_act_features(user_response))
    if (clas != 'Bye'):
        if (clas == 'Emotion'):
            flag = False
            prYellow("Chatter: You are welcome..")
        else:
            if (greeting(user_response) != None):
                print("\033[93m {}\033[00m".format("Chatter: " + greeting(user_response)))
            else:
                print("\033[93m {}\033[00m".format("Chatter: ", end=""))
                res = (response(user_response))
                prYellow(res)
                sent_tokens.remove(user_response)
                tts = gTTS(text=res, lang="en", tld="com")
                tts.save("file.mp3")
                os.system("mpg123 file.mp3")
    else:
        flag = False
        prYellow("User: Bye! take care..")
