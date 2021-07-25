# -*- coding: utf-8 -*-

# Student Name: Tommy Kangdra
# Student ID: A0218866N

# version:
# python: 2.7.11
# pyspark: 2.2.1
# mvn: 3.5.2
# scala: 2.12.4

# STEP 0: importing the relevant library
import pyspark
from pyspark import SparkContext, SparkConf
import random
import sys
import os
import math

# initiating spark conference and spark context
conf = SparkConf().setAppName("Lab2").setMaster("local[*]")
sc = SparkContext().getOrCreate(conf)

# functions


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


def tf_idf_func(inputs):
    # STEP 3 function: compute the tf-idf
    N = 35
    DF = len(inputs[1])
    result = []
    for i in inputs[1]:
        tf = i[1]
        tf_idf = (1 + math.log10(i[1])) * (math.log10(float(N)/DF))
        result.append((i[0], (inputs[0], tf_idf)))
    return result


def normalized_s(inputs):
    # STEP 4 function: compute the normalized tf_idf
    results = []
    s = 0
    for i in inputs[1]:
        s += i[1]**2
    for i in inputs[1]:
        norm = i[1]/s**0.5
        results.append(((i[0], inputs[0]), norm))
    return results


def mag(x):
    # function to calculate the magnitude of vector
    return math.sqrt(sum(i**2 for i in x))


def dot(x, y):
    # function to calculate dot product
    return sum(x_i*y_i for x_i, y_i in zip(x, y))


# STEP 0: load the data
total_docs = 35
query = sc.textFile('query.txt')
stopwords = sc.textFile('stopwords.txt').collect()
path = './datafiles/'
rdd2 = sc.parallelize([])
for n in range(1, total_docs+1):
    file = sc.textFile(path + 'f' + str(n) + '.txt')
    # STEP 1: Preprocess the document
    preprocessed = file.map(preprocess)
    words = preprocessed.flatMap(lambda l: l.split())
    word_doc = words.map(lambda x: ('f' + str(n), x))
    rdd2 = rdd2.union(word_doc)

# STEP 2 Compute term ferequency of every word in document
pair = rdd2.map(lambda x: (x, 1))
counts = pair.reduceByKey(lambda n1, n2: n1 + n2)

# remapping the result to ((document, word), count_value)
remap_kv = counts.map(lambda x: (
    x[0][1], (x[0][0], x[1]))).groupByKey().mapValues(list)

# STEP 3
tf_idf = remap_kv.map(tf_idf_func).flatMap(
    lambda x: x).groupByKey().mapValues(list)

# STEP 4
normalized = tf_idf.map(normalized_s).flatMap(lambda x: x)

# STEP PRE 5: preprocess before computing relevance
# PRE5A: cartesian all words and documents
all_words = rdd2.map(lambda x: x[1]).distinct().sortBy(
    lambda x: x, ascending=True, numPartitions=4)
all_docs = sc.parallelize(['f' + str(n) for n in range(1, total_docs+1)])
all_words_docs = all_words.cartesian(all_docs).map(lambda x: ((x[0], x[1]), 0))
# union the document with STEP 4 normalized
all_words_normalized = (
    normalized + all_words_docs).reduceByKey(lambda n1, n2: n1+n2, numPartitions=40)
sorted_words_normalized = all_words_normalized.map(lambda x: (
    x[0][1], (x[0][0], x[1]))).sortBy(lambda x: str(x[0] + x[1][0]), ascending=True)

# PRE5B: make a (k,v) where k is the document (e.g. 'f1') v is the matrix of the values
docs_vectors = sorted_words_normalized.map(
    lambda x: (x[0], x[1][1])).groupByKey().mapValues(list)
# collecting documents and the matrix of tf_idf_normalized to process in step 5
documents_vectors = docs_vectors.collect()

# PRE5C: make a vector of the query words with all words from all documents (0 & 1 Matrices)
query_vector = query.flatMap(lambda x: x.split()).map(lambda x: (x, 1))
all_words_vector = all_words.map(lambda x: (x, 0))
query_vector = query_vector.union(
    all_words_vector).reduceByKey(lambda n1, n2: n1+n2)
matrix = query_vector.sortBy(
    lambda x: x[0], ascending='True').map(lambda x: x[1])
# collecting all words and the matrix of the query to process in step 5
query_vector_list = matrix.collect()

# STEP 5 Compute relevance of each document w.r.t a query
result = []
for i in range(len(documents_vectors)):
    doc = documents_vectors[i]
    document_vector = doc[1]
    relevance = dot(document_vector, query_vector_list) / \
        (mag(document_vector) * mag(query_vector_list))
    result.append((doc[0], relevance))
# sort in descending order according to relevance value
result.sort(key=lambda x: x[1], reverse=True)


# output the files of top k relevance documents
k = 5
outfile = open("./final_output_taskB.txt", "w")
for l in result[0:k]:
    outfile.writelines('<' + str(l[0]) + '> ' + '<' + str(l[1]) + '>\n')
outfile.close()

sc.stop()
