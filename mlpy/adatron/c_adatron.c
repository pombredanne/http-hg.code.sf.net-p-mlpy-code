#include <stdlib.h>
#include <float.h>
#include <math.h>

#define MIN( A , B ) ((A) < (B) ? (A) : (B))
#define MAX( A , B ) ((A) > (B) ? (A) : (B))


int adatron(long *y, double *K, int n, double C, int maxsteps, double eps,
	    double *alpha, double *margin)
{
  double *z = NULL;
  double delta;
  int i, j, k;
  double zplus, zminus;
  int nplus, nminus;
  

  z = (double *) malloc (n * sizeof(double));
  
  for (i=0; i<maxsteps; i++)
    {
      for (j=0; j<n; j++)
	{
	  z[j] = 0.0;
	  for (k=0; k<n; k++)
	    /* z_j = sum(alpha_k y_k K_kj) */
	    z[j] += alpha[k] * y[k] * K[j + (k * n)];
	  delta = (1 - (y[j] * z[j])) / K[j + (j * n)];
	  alpha[j] = MIN(MAX(0, alpha[j]+delta), C);  
	}

      /* margin */
      zplus = DBL_MAX; zminus = -DBL_MAX;
      nplus = 0; nminus = 0;
      for (k=0; k<n; k++)
	{
	  if ((y[k]==+1) && (alpha[k]<C))
	    {
	      zplus = MIN(zplus, z[k]);
	      nplus++;
	    }
	  if ((y[k]==-1) && (alpha[k]<C))
	    {
	      zminus = MAX(zminus, z[k]);
	      nminus++;
	    }
	}

      if ((nplus == 0) || (nminus == 0))
	*margin = 0.0;
      else
	*margin = 0.5 * (zplus - zminus);

      if (fabs(1.0 - *margin) < eps)
	break;
    }
  
  free(z);
  return i;
}
