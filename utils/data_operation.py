from __future__ import division
import math

""" Calculates the l2 distance between two vectors """
def eucliden_distance(x1,x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow( x1[i] - x2[i],2)
    return math.sqrt(distance)
