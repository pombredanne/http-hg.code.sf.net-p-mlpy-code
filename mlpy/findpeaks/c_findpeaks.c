/*  
    This code is written by Davide Albanese <davide.albanese@gmail.com>.
    (C) Davide Albanese.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>


#define MAX(a, b) ((a) > (b) ? (a):(b))
#define MIN(a, b) ((a) < (b) ? (a):(b))


struct node
{
  long data;
  struct node *next;
} Node;

typedef struct node NODE;


int
enqueue(NODE **head, NODE **tail, double data)
{
  NODE *tmp;
  tmp = malloc(sizeof(NODE));
  
  if (tmp == NULL)
    return 1;
  
  tmp->data = data;
  tmp->next = NULL;
  
  if (*head == NULL)
    *head = tmp;
  else
    (*tail)->next = tmp;
  *tail = tmp;
  
  return 0;
}


static PyObject *
findpeaks_findpeaks_win(PyObject *self, PyObject *args, PyObject *keywds)
{
  /* input */
  PyObject *x = NULL;
  int span;

  /* x contiguous */
  PyObject *xCont = NULL;
  
  double * xC;
  npy_intp xDim;
  
  static char *kwlist[] = {"x", "span", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "Oi", kwlist, &x, &span))
    return NULL;

  xCont = PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY);
  if (xCont == NULL) return NULL;

  /* Check number of dimension */
  if (PyArray_NDIM(xCont) != 1)
    {
      PyErr_SetString(PyExc_ValueError, "x must be 1-dimensional");
      return NULL;
    }
  
  /* Check span */
  if ((span % 2 == 0) || (span < 3))
    {
       PyErr_SetString(PyExc_ValueError, "span must be >= 3 and odd");
       return NULL;
    }
  
  xC = (double *) PyArray_DATA(xCont);
  xDim = PyArray_DIM(xCont, 0);

  /*** Start alghorithm ***/
  
  NODE *head = NULL, *tail = NULL;
  npy_intp l_min, l_max, r_min, r_max;
  npy_intp i, j;
  npy_intp m;
  short int is_peak;
  int dist;

  dist = (span + 1) / 2;

  m = 0;
  for (i=0; i<xDim; i++)
    {
      l_min = MAX(i-dist+1, 0);
      l_max = i-1;
      r_min = i+1;
      r_max = MIN(i+dist-1, xDim-1);
      
      is_peak = 1;
      
      /* left side */
      for (j=l_min; j<=l_max; j++)
	if (xC[j] >= xC[i])
	  {
	    is_peak = 0;
	    break;
	  }
    
      /* right side */
      if (is_peak == 1)
	for (j=r_min; j<=r_max; j++)
	  if (xC[j] >= xC[i])
	    {
	      is_peak = 0;
	      break;
	    }
          
      if (is_peak == 1)
	{
	  if (enqueue(&head, &tail, i))
	    return NULL; 
	  m++;
	}
    }

  /*** End alghorithm ***/

  /*** Start build the output array ***/

  npy_intp retDims[1];
  PyObject *retCont = NULL;
  long *retC;

  NODE *tmp = NULL;
  npy_intp k;
    
  retDims[0] = m;
  retCont = PyArray_SimpleNew(1, retDims, NPY_LONG);
  retC = (long *) PyArray_DATA(retCont);
    
  /* copy and free the list */
  for (k=0; k<m; k++)
    {
      tmp = head->next;
      retC[k] = head->data;
      free(head);
      head = tmp;
    }
  tail = NULL;

  /*** End build the output array ***/ 

  Py_DECREF(xCont);
  return Py_BuildValue("N", retCont);
}


static char module_doc[] = "";
static char findpeaks_win_doc[] = "";


/* Method table */
static PyMethodDef findpeaks_methods[] = {
  {"findpeaks_win",
   (PyCFunction)findpeaks_findpeaks_win,
   METH_VARARGS | METH_KEYWORDS,
   findpeaks_win_doc},
  {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cfindpeaks",
    module_doc,
    -1,
    findpeaks_methods,
    NULL, NULL, NULL, NULL
  };
#endif

void initc_findpeaks(void)
{
  Py_InitModule3("c_findpeaks", findpeaks_methods, module_doc);
  import_array();
}
