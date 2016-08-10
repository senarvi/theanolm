'''
Created on Aug 10, 2016

@author: senarvi
'''

import math
from decimal import *

def interpolate_linear(logprob1, logprob2, weight1):
    """Performs linear interpolation of two probabilities.

    :type logprob1: float
    :param logprob1: logarithm of first input probability

    :type logprob2: float
    :param logprob2: logarithm of second input probability

    :type weight1: float
    :param weight1: interpolation weight for the first probability

    :rtype: float
    :returns: logarithm of the weighted sum of the input probabilities 
    """

    prob1 = math.exp(logprob1)
    prob2 = math.exp(logprob2)
    if (prob1 > 0) and (prob2 > 0):
        weight2 = 1.0 - weight1
        prob = weight1 * prob1
        prob += weight2 * prob2
        return math.log(prob)
    else:
        # An exp() resulted in an underflow. Use the decimal library.
        getcontext().prec = 16
        d_weight1 = Decimal(weight1)
        d_weight2 = Decimal(1.0) - d_weight1
        d_logprob1 = Decimal(logprob1)
        d_logprob2 = Decimal(logprob2)
        d_prob = d_weight1 * d_logprob1.exp()
        d_prob += d_weight2 * d_logprob2.exp()
        return float(d_prob.ln())

def interpolate_loglinear(logprob1, logprob2, prior1, prior2):
    """Performs log-linear interpolation of two probabilities.

    This is not real log-linear interpolation, because we don't normalize the
    result. Thus the result is not a real probability.

    :type logprob1: float
    :param logprob1: first input log probability

    :type logprob2: float
    :param logprob2: second input log probability

    :type prior1: float
    :param prior1: weight for the first log probability

    :type prior2: float
    :param prior2: weight for the second log probability

    :rtype: float
    :returns: weighted sum of the input log probabilities 
    """

    numerator = prior1 * logprob1;
    numerator += prior2 * logprob2;
    return numerator
