# version
# python: 2.7.11
# pyspark: 2.2.1
# mvn: 3.5.2
# scala: 2.12.4

# Student Name: Tommy Kangdra
# Student ID: A0218866N

# STEP 0: importing the relevant library
import pyspark
from pyspark import SparkConf, SparkContext
import random
import sys
import os

# initiating spark conference and spark context
conf = SparkConf().setAppName("Lab2").setMaster("local[*]")
sc = SparkContext().getOrCreate(conf)

# function


def preprocess(line):
    # Step 1 function: to preprocess the data
    # Step 1a: convert all then words to lower case
    line = line.lower()
    words = line.split()
    sentence_transform = []

    for word in words:
        # Step 1b: remove the stop words
        if (word not in stopwords):
            word_transform = ""
            for character in word:
                # Step 1c: drop symbols (only take in the alphanumeric character)
                if character.isalnum():
                    word_transform += character
            # Step1d: Drop independent numbers
            if not(word_transform.isdigit()):
                sentence_transform.append(word_transform)
    return ' '.join(sentence_transform)


def word_pairing(line):
    # Step 2 function: to make word pair
    import itertools

    words = line.split()
    word_pair = []
    if len(words) > 1:
        # Step 2a for loop to make word pair from the document
        for n in range(len(line)):
            if n <= len(words) - 2:
                # if function to sort the word pair in ascending manner
                if words[n] <= words[n+1]:
                    word_pair.append(((words[n], words[n+1]), 1))
                else:
                    word_pair.append(((words[n+1], words[n]), 1))
                # (k,v) where v = 1 to count the numbers of the wordpair
    # Step 2b for loop to make cartesian product with itertools.product
    other_pair = itertools.product(words, words)
    for pairs in other_pair:
        if pairs[0] != pairs[1]:
            if pairs[0] > pairs[1]:
                a, b = pairs
                pairs = (b, a)
            word_pair.append((pairs, 0))
            # (k,v) where v=0 so as not to mess the count of the wordpair
    return word_pair


# using for loops to load the files and union it together
# and load the stopwords
path = './datafiles/'
rdd1 = sc.parallelize([])
for n in range(1, 36):
    file = sc.textFile(path + 'f' + str(n) + '.txt')
    rdd1 = rdd1.union(file)
stopwords = sc.textFile('stopwords.txt').collect()


# STEP 1 preprocess the documents
preprocessed = rdd1.map(preprocess).filter(lambda x: x != '')

# STEP 2
# STEP 2A: Generate the word pair with (k,v,1) for adjacent word pair
# and generate (k,v,0) for all other wordpair
word_pair = preprocessed.map(word_pairing)
word_pair = word_pair.flatMap(lambda x: x)

# Step 2B to compute the count by adding the value
count_pairs = word_pair.reduceByKey(lambda n1, n2: n1 + n2)

# Step 3c Sort the list of word pairs in descending order and obtain top-5
top5 = count_pairs.sortBy(lambda x: x[1], ascending=False).take(5)

# writing the list to the text
outfile = open("./final_output_taskA.txt", "w")
for l in top5:
    outfile.writelines('<' + str(l[0]) + '> ' + '<' + str(l[1]) + '>\n')
outfile.close()

sc.stop()
