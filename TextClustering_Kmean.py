import nltk
import collections
import numpy as np
from sklearn.cluster import KMeans
from nltk import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import k_means
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


"""nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')"""


sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "The rain in Spain stays mainly in the plain.",
    "A journey of a thousand miles begins with a single step.",
    "Better late than never.",
    "Actions speak louder than words.",
    "Every cloud has a silver lining.",
    "Don't count your chickens before they hatch.",
    "Birds of a feather flock together.",
    "A picture is worth a thousand words.",
    "The pen is mightier than the sword.",
    "You can't judge a book by its cover.",
    "A watched pot never boils.",
    "A stitch in time saves nine.",
    "Curiosity killed the cat.",
    "Ignorance is bliss.",
    "Knowledge is power.",
    "Practice makes perfect.",
    "Rome wasn't built in a day.",
    "The early bird catches the worm.",
    "When in Rome, do as the Romans do.",
    "Where there's smoke, there's fire.",
    "You reap what you sow.",
    "You can't have your cake and eat it too.",
    "A penny saved is a penny earned.",
    "Absence makes the heart grow fonder.",
    "An apple a day keeps the doctor away.",
    "Beauty is in the eye of the beholder.",
    "Beggars can't be choosers.",
    "Better safe than sorry.",
    "Blood is thicker than water.",
    "Cleanliness is next to godliness.",
    "Don't bite the hand that feeds you.",
    "Don't put all your eggs in one basket.",
    "Easy come, easy go.",
    "Fortune favors the brave.",
    "Good things come to those who wait.",
    "Honesty is the best policy.",
    "If it ain't broke, don't fix it.",
    "If you can't beat them, join them.",
    "It takes two to tango.",
    "Keep your friends close and your enemies closer.",
    "Laughter is the best medicine.",
    "Let bygones be bygones.",
    "Look before you leap.",
    "Money doesn't grow on trees.",
    "Necessity is the mother of invention.",
    "No pain, no gain.",
    "Out of sight, out of mind.",
    "Patience is a virtue.",
    "Silence is golden.",
    "The best things in life are free.",
    "The grass is always greener on the other side.",
    "Time flies when you're having fun.",
    "Two heads are better than one.",
    "United we stand, divided we fall.",
    "What goes around, comes around.",
    "When the going gets tough, the tough get going.",
    "Where there's a will, there's a way.",
    "You can't make an omelet without breaking a few eggs.",
    "You can't teach an old dog new tricks.",
    "A friend in need is a friend indeed.",
    "A leopard can't change its spots.",
    "A rolling stone gathers no moss.",
    "Actions have consequences.",
    "All good things must come to an end.",
    "All that glitters is not gold.",
    "All's well that ends well.",
    "As you sow, so shall you reap.",
    "Barking dogs seldom bite.",
    "Begging the question.",
    "Birds of a feather flock together.",
    "Burn the midnight oil.",
    "Castles in the air.",
    "Cat got your tongue?",
    "Cry over spilled milk.",
    "Don't beat around the bush.",
    "Don't cry wolf.",
    "Don't jump to conclusions.",
    "Fools rush in where angels fear to tread.",
    "Get out of my hair.",
    "Give someone the cold shoulder.",
    "Go the extra mile.",
    "Grasping at straws.",
    "Have your cake and eat it too.",
    "Hit the nail on the head.",
    "Jump on the bandwagon.",
    "Keep your chin up.",
    "Let the cat out of the bag.",
    "Light at the end of the tunnel.",
    "Make hay while the sun shines.",
    "Miss the boat.",
    "Not the sharpest tool in the shed.",
    "Off the beaten path.",
    "On cloud nine.",
    "Play it by ear.",
    "Pull someone's leg.",
    "Put your best foot forward.",
    "Raining cats and dogs.",
    "Run of the mill."
]


def tokenize(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stem = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return stem

vec_ = TfidfVectorizer(tokenizer=tokenize, stop_words=stopwords.words('english') , lowercase=True)
new_ = vec_.fit_transform(sentences)
print((new_*new_.T).toarray().shape)
print(new_.shape)

test_sentences = [
    "The cat sat on the mat.",
    "She baked a cake for her friend's birthday.",
    "The stars shine brightly in the night sky.",
    "He jogs in the park every morning.",
    "The car stopped at the traffic light.",
    "They enjoyed a picnic by the lake.",
    "The dog barked at the mailman.",
    "She painted a beautiful landscape.",
    "The children played games in the backyard.",
    "He read a book under the tree.",
    "The coffee shop was crowded with customers.",
    "She knitted a scarf for the winter.",
    "The flowers bloomed in the garden.",
    "He listened to his favorite music.",
    "The airplane took off smoothly.",
    "They watched a movie together.",
    "The sun set over the horizon.",
    "She wrote a letter to her friend.",
    "The rain poured down heavily.",
    "He built a model airplane."
]
new__ = vec_.transform(test_sentences)

est = KMeans(10)
est.fit(new_)
clusters = collections.defaultdict(list)
for i, label in enumerate(est.labels_):
    clusters[label].append(i)
for cluster in range(10):
    print("CLUSTER ", cluster, ":")
    for i, sentence in enumerate(clusters[cluster]):
        print("\tSENTENCE ", i, ": ", sentences[sentence])

pres = est.predict(new__)
print("PRES ", pres)

#for visualisation purposes we convert spare array into and 2d array with the help of PCA
