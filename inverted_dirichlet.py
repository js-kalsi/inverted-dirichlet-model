"""
/*
*       Coded by : Jaspreet Singh Kalsi.
*
*       "Thesis  Chapter-2 Part A
*       (Image Fragmentation using Inverted Dirichlet Distribution using Markov Random Field as a Prior).
*
*       ```python core.py <Image-Name>```
*
*/

"""
import sys
from numpy import sum as SUM
from numpy import zeros as ZEROS
from scipy.special import gammaln as GAMMALN
from numpy import log as LOG
from numpy import exp as EXP
from numpy import asarray as ASARRAY

"""
/**
 * This function add the array's element and return them in the form of a String.
 * @param  {Integer} a.
 * @return {String} which contains the Sum of Array.
 */
"""

class inverted_dirichlet:

    def __init__(self, M, alpha, data, dim):
        self.M = M
        self.alpha = alpha
        self.data = data
        self.dim = dim
    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    def pdf_fetcher(self):
        probability = ZEROS((len(self.data), self.M))
        for a_i, a_v in enumerate(self.alpha):
            for p_i, d_v in enumerate(self.data):
                probability[p_i][a_i] = self.pdf(ASARRAY(d_v).reshape(1, self.dim), ASARRAY(a_v).reshape(1, self.dim + 1))
        return probability


    """
    /**
     * This function add the array's element and return them in the form of a String.
     * @param  {Integer} a.
     * @return {String} which contains the Sum of Array.
     */
    """
    @staticmethod
    def pdf(d_v, a_v):
        return EXP(GAMMALN(SUM(a_v)) - SUM(GAMMALN(a_v) )
                   + SUM((a_v[:, :-1] - 1) * LOG(d_v)) - (SUM(a_v) * LOG(1 + SUM(d_v))))
