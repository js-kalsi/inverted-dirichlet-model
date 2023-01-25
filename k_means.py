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

from sklearn.cluster import KMeans
from numpy import ndarray

class KMeans:

    def __init__(self, img_pixels: ndarray, no_of_clusters: int):
        self.img_pixels = img_pixels
        self.KM = KMeans(n_clusters=no_of_clusters).fit(self.imgPixels)

    def predict(self):
        return self.KM.predict(self.imgPixels)
