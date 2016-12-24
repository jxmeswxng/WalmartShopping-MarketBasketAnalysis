'''
INVESTIGATING WALMART SHOPPING TRIPS WITH MARKET-BASKET ANALYSIS

CIS192 Final Project
James Wang
PennID: 46576241
Date: 12/19/16
'''

import os
import time
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
###### Preparing the data
###############################################################################

### import data
raw_data = pd.read_csv("trip_data.csv")

### pre-processing

# remove returns: ScanCounts < 0
data = raw_data[raw_data["ScanCount"] > 0]

# recode Weekdays as integers
data["Weekday"][data["Weekday"] == "Monday"] = 1
data["Weekday"][data["Weekday"] == "Tuesday"] = 2
data["Weekday"][data["Weekday"] == "Wednesday"] = 3
data["Weekday"][data["Weekday"] == "Thursday"] = 4
data["Weekday"][data["Weekday"] == "Friday"] = 5
data["Weekday"][data["Weekday"] == "Saturday"] = 6
data["Weekday"][data["Weekday"] == "Sunday"] = 7

# drop NaNs from "Upc" column
data = data[np.isfinite(data["Upc"])]

### peek data
data.shape
data.head(10)
data.columns.values



###############################################################################
###### Plotting the data
###############################################################################

### Single-Level Plots
# TripType
x_labels = list(data.TripType.value_counts().index)
x = range(len(x_labels))
y = data.TripType.value_counts()
plt.figure(figsize=(10, 5))
plt.bar(x, y, align="center")
plt.xticks(x, x_labels, rotation=90)
plt.title("Frequency of All Trip Types")

# VisitNumber
len(np.unique(data.VisitNumber))

# Weekday
x_labels = list(data.Weekday.value_counts().index)
y = data.Weekday.value_counts()
plt.figure(figsize=(10, 5))
plt.bar(x_labels, y, align="center")
plt.title("Frequency of Trips by Weekday")

# Upc
data.Upc.value_counts()
len(np.unique(data.Upc))

# ScanCount
data.ScanCount.value_counts()

# DepartmentDescription
x_labels = list(data.DepartmentDescription.value_counts().index)
x = np.arange(0, len(x_labels) * 2, 2)
y = data.DepartmentDescription.value_counts()
plt.figure(figsize=(20, 10))
plt.bar(x, y, align="center")
plt.xticks(x, x_labels, rotation=90)
plt.title("Frequency of Departments")

# FinelineNumber
data.FinelineNumber.value_counts()


### Cross-Level Plots
# By TripType: Counts by Weekday and Counts by DepartmentDescription
triptypes = np.unique(data.TripType)
tvd_measures = {}
for i in triptypes:
    # upper plot
    subset = data[data["TripType"] == i]["Weekday"]
    x = list(subset.value_counts().index)
    y = subset.value_counts()
    plt.figure()
    plt.subplot(211)
    plt.bar(x, y, align="center")
    plt.title(i)
    plt.subplots_adjust(hspace=0.4)
    # calculate TVD of frequencies compared to uniform distribution
    num_missing = list(set(range(1, 8)) - set(x))
    x_chi = x + num_missing                            # fill in incomplete values
    y_chi = np.array(list(y) + [0] * len(num_missing)) # fill in incomplete values
    obs_dist = np.array((x_chi, y_chi)).T
    obs_dist = obs_dist[obs_dist[:, 0].argsort()][:, 1]
    obs_freq = obs_dist / float(sum(obs_dist))
    exp_freq = np.array([1.0 / 7] * 7)
    tvd_measures[i] = sum(abs(obs_freq - exp_freq)) / 2

    # lower plot
    subset = data[data["TripType"] == i]["DepartmentDescription"]
    y = subset.value_counts()[:5]
    labels = list(y.index)
    x = range(5)
    plt.subplot(212)
    plt.bar(x, y, align="edge")
    plt.xticks(x, labels, rotation=45)
    plt.title(i)

# By Weekday: Most frequent DepartmentDescription and TripType
weekdays = np.unique(data.Weekday)
for i in weekdays:
    subset = data[data["Weekday"] == i]["TripType"]
    x_labels = list(subset.value_counts().index)
    x = range(len(x_labels))
    y = subset.value_counts()
    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.bar(x, y, align="center")
    plt.xticks(x, x_labels, rotation=90)
    plt.title(i)
    plt.subplots_adjust(hspace=0.5)

    subset = data[data["Weekday"] == i]["DepartmentDescription"]
    y = subset.value_counts()[:5]
    labels = list(y.index)
    x = range(5)
    plt.subplot(212)
    plt.bar(x, y, align="edge")
    plt.xticks(x, labels, rotation=45)
    plt.title(i)


### Compare TVDs of aggregate weekly frequencies to trip type weekly frequencies
tvd_array = np.array([list(i) for i in tvd_measures.items()])
tvd_array = tvd_array[tvd_array[:, 1].argsort()]
tvd_array[:5]
tvd_array[-5:]
pd.DataFrame(tvd_array.T).to_csv("tvd_weekdays.csv", index=False, header=False)



###############################################################################
###### Market-Basket Analysis
###############################################################################

### setup
baskets = np.unique(data.VisitNumber)
baskets_dd = defaultdict(set)
mb_subset = data[["VisitNumber", "Upc"]]


### create baskets
for i in baskets: # takes about 3 minutes
    baskets_dd[i] = baskets_dd[i].union(set(mb_subset[mb_subset["VisitNumber"] == i]["Upc"]))


### distribution of basket size
basket_sizes = [len(baskets_dd[i]) for i in baskets_dd]
pd.DataFrame(basket_sizes).describe()


### distribution of item frequency
item_freq = mb_subset.groupby(by="Upc").count()
pd.DataFrame(item_freq).describe()


### finding frequent itemsets: A-Priori Algorithm
baskets = baskets_dd
max_itemset_size = 10
'''
# choosing support threshold
sum(np.array(item_freq["VisitNumber"]) > 10)   # 11696, too many
sum(np.array(item_freq["VisitNumber"]) > 50)   #  1506, <- looks like a good number
sum(np.array(item_freq["VisitNumber"]) > 100)  #   529, too few
sum(np.array(item_freq["VisitNumber"]) > 500)  #    35, too few
sum(np.array(item_freq["VisitNumber"]) > 1000) #    11, too few
'''
support_threshold = 50
freq_itemset_by_size = {}

# find itemsets of size 1
itemset_1 = defaultdict(int)
for i in baskets.values():
    for j in i:
        itemset_1[j] += 1
freq_itemset_by_size[1] = {frozenset([i]): j for i, j in itemset_1.items() if j > support_threshold}

# fine itemsets of size > 1, takes about 3 minutes
for i in range(2, max_itemset_size + 1):
    itemset_i = defaultdict(int)
    k_minus_1_itemsets = freq_itemset_by_size[i - 1]
    for j in baskets.values():
        for k in k_minus_1_itemsets:
            if k.issubset(j):
                for l in (j - k):
                    set_l = set([l])
                    union = k.union(set_l)
                    itemset_i[union] += 1
    freq_itemset_by_size[i] = {key: value for key, value in itemset_i.items() if value > support_threshold}

# keep itemsets of size != 1 and n != 0
freq_itemset_by_size = {i: j for i, j in freq_itemset_by_size.items() if i != 1 and len(j) != 0}


### generate association rules
rules = [(j - set([k]), k) for i in freq_itemset_by_size for j in freq_itemset_by_size[i] for k in j]


### calculate counts, confidence, and interest of rules
# counts
counts_cond_and_result = defaultdict(int)
counts_cond = defaultdict(int)
for basket in baskets.values(): # takes about 3 minutes
    for condition, result in rules:
        if condition.issubset(basket):
            counts_cond[(condition, result)] += 1
            if result in basket:
                counts_cond_and_result[(condition, result)] += 1

# confidence
confidence_rules = {i: counts_cond_and_result[i] / float(counts_cond[i])
                    for i in rules}

# interest
n = sum(itemset_1.values())
interest_rules = {i: confidence_rules[i] - (itemset_1[i[1]] / float(n))
                  for i in rules}


### identify most confident and most interesting rules
confidence_rules_sorted = sorted(confidence_rules.items(), key=lambda x: x[1], reverse=True)
interest_rules_sorted = sorted(interest_rules.items(), key=lambda x: x[1], reverse=True)

threshold = 0.9
confidence_rules_top = [i for i in confidence_rules_sorted if i[1] > threshold]
interest_rules_top = [i for i in interest_rules_sorted if i[1] > threshold]

confidence_and_interest_rules = {i[0]: i[1] + j[1] for i in confidence_rules_sorted
                                 for j in interest_rules_sorted
                                 if i[0] == j[0] and
                                 i[1] > threshold and
                                 j[1] > threshold}
confidence_and_interest_rules_top = sorted(confidence_and_interest_rules.items(), key=lambda x: x[1], reverse=True)


### format results
for i, j in confidence_rules_top:
    print("Given {}, then {}. Confidence: {}".format(list(i[0])[0], i[1], j))

for i, j in interest_rules_top:
    print("Given {}, then {}. Interest: {}".format(list(i[0])[0], i[1], j))

for i, j in confidence_and_interest_rules_top:
    print("Given {}, then {}. Combined: {}".format(list(i[0])[0], i[1], j))


### identify departments based on highest combined rules
items_dpmts = data[["Upc", "DepartmentDescription"]]
items_dpmts_map = {i: j for i, j in zip(items_dpmts["Upc"], items_dpmts["DepartmentDescription"])}

for i, j in confidence_and_interest_rules_top:
    if len(list(i[0])) == 1:
        print("Given {}, then {}. Combined: {}".format(items_dpmts_map[list(i[0])[0]], items_dpmts_map[i[1]], j))
    if len(list(i[0])) == 2:
        print("Given {} and {}, then {}. Combined: {}".format(items_dpmts_map[list(i[0])[0]], items_dpmts_map[list(i[0])[1]], items_dpmts_map[i[1]], j))
    if len(list(i[0])) == 3:
        print("Given {}, {}, and {}, then {}. Combined: {}".format(items_dpmts_map[list(i[0])[0]], items_dpmts_map[list(i[0])[1]], items_dpmts_map[list(i[0])[2]],  items_dpmts_map[i[1]], j))
    if len(list(i[0])) == 4:
        print("Given {}, {}, {}, and {}, then {}. Combined: {}".format(items_dpmts_map[list(i[0])[0]], items_dpmts_map[list(i[0])[1]], items_dpmts_map[list(i[0])[2]], items_dpmts_map[list(i[0])[3]],  items_dpmts_map[i[1]], j))
