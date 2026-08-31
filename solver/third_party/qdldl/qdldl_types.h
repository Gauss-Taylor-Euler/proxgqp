/*
 * Standalone type bindings for QDLDL.
 *
 * The copy of QDLDL that ships inside OSQP's codegen tree has this header
 * rewired to include "osqp_api_types.h" and take OSQPInt / OSQPFloat.  This
 * replaces that with the plain types, so qdldl.c and qdldl.h are UPSTREAM AND
 * UNMODIFIED and nothing here pulls in OSQP.  Only the typedefs are ours.
 *
 * QDLDL itself is Copyright 2018 Paul Goulart, Bartolomeo Stellato, Goran
 * Banjac and the OSQP developers, Apache License 2.0; see LICENSE beside this
 * file and the header of qdldl.c.
 */
#ifndef QDLDL_TYPES_H
#define QDLDL_TYPES_H

#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int    QDLDL_int;      /* indices; matches Eigen's default StorageIndex */
typedef double QDLDL_float;
typedef int    QDLDL_bool;     /* int, not bool: QDLDL is C89 */

#define QDLDL_INT_MAX INT_MAX

#ifdef __cplusplus
}
#endif

#endif /* QDLDL_TYPES_H */
