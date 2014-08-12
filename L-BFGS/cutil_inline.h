#ifndef CUTIL_INLINE_H
#define CUTIL_INLINE_H

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef __max
#define __max(a, b) ((a) > (b) ? a : b)
#endif

#ifndef __min
#define __min(a, b) ((a) < (b) ? a : b)
#endif

#define cutilSafeCall(x) {if(!cudaSuccess == x) exit(0);}

#endif
