from version import v as __version__
import sys

# extension modules
import gsl
from libsvm import *
from liblinear import *
from libml import *
from findpeaks import *
from kmeans import *
from kernel import *
from adatron import *
from canberra import *

# python modules
from crossval import *
from hcluster import *
from metrics import *
from perceptron import *
from da import *
from ols import *
from ridge import *
from bordacount import *
from lars import *
from elasticnet import *
from dimred import *
from irelief import *
from parzen import *
from confidence import *

import crossval
import hcluster
import metrics
import perceptron
import da
import ols
import ridge
import bordacount
import lars
import elasticnet
import dimred
import irelief
import parzen
import confidence

# visible submodules
import wavelet


__all__ = []
__all__ += crossval.__all__
__all__ += hcluster.__all__
__all__ += metrics.__all__
__all__ += perceptron.__all__
__all__ += da.__all__
__all__ += ols.__all__
__all__ += ridge.__all__
__all__ += bordacount.__all__
__all__ += lars.__all__
__all__ += elasticnet.__all__
__all__ += dimred.__all__
__all__ += irelief.__all__
__all__ += parzen.__all__
__all__ += confidence.__all__
