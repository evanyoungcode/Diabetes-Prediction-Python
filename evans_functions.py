#Functions I wrote to clean up the code and I would reuse for SLR and kNN Regression




#Imputes zeros with the mean of the non zeros
def impute_zeros_with_mean(data_list):

#returns and prints if no zeros in the data_list
    if 0 not in data_list:
        print("There is no zeros to impute in the list")
        return data_list
    
#putting all numbers not needed to be imputed in a list
    not_zero_list = []
    for x in data_list:
        if x != 0:
            not_zero_list.append(x)
        
#calculating the mean of non zeros
    mean = sum(not_zero_list) / len(not_zero_list)

#replacing the zeros with the mean
    data_list_amputed = []
    for x in data_list:
        if x == 0:
            data_list_amputed.append(mean)
        else:
            data_list_amputed.append(x)
    return data_list_amputed




#Calculates the RMSE(RMSE = squareroot of (SSE/Data Points)
def rmse(sse, x):
    i = len(x)
    return (sse / i) ** 0.5

#taken from working_with_data since the file is filled with graphs
from typing import List, Tuple
from scratch.linear_algebra import Vector, vector_mean
from scratch.statistics import standard_deviation



def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """returns the means and standard deviations for each position"""
    dim = len(data[0])

    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]

    return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)
assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

def rescale(data: List[Vector]) -> List[Vector]:
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position
    as is if its standard deviation is 0.)
    """
    dim = len(data[0])
    means, stdevs = scale(data)

    # Make a copy of each vector
    rescaled = [v[:] for v in data]

    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]

    return rescaled



from scratch.linear_algebra import Matrix, Vector, make_matrix
from scratch.statistics import correlation

def correlation_matrix(data: List[Vector]) -> Matrix:
    """
    Returns the len(data) x len(data) matrix whose (i, j)-th entry
    is the correlation between data[i] and data[j]
    """
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])

    return make_matrix(len(data), len(data), correlation_ij)