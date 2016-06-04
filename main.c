#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>
#include <mpi.h>

#define PREC 100
#define NLIMBS \
    ((mp_size_t) ((__GMP_MAX (53, PREC) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define PACKED_LIMBS ((size_t) NLIMBS + 3)
#define PACKED_BYTES ((size_t) PACKED_LIMBS * sizeof(mp_limb_t))

mp_limb_t *pack(mp_limb_t *dest, mpf_t* src, int n)
{
    if (dest == NULL)
        dest = (mp_limb_t *)malloc(PACKED_BYTES * n);
    size_t offset = 0;
    for (int i = 0; i < n; i++) {
        dest[offset + 0] = src[i]->_mp_size;
        dest[offset + 1] = src[i]->_mp_prec;
        dest[offset + 2] = src[i]->_mp_exp;
        memcpy(&dest[offset + 3], src[i]->_mp_d, NLIMBS * sizeof(mp_limb_t));
        offset += PACKED_LIMBS;
    }
    return dest;
}

mpf_t *unpack(mpf_t *dest, mp_limb_t *src, int n)
{
    if (dest == NULL)
        dest = (mpf_t *)malloc(sizeof(mpf_t) * n);
    size_t offset = 0;
    for (int i = 0; i < n; i++) {
        dest[i]->_mp_size = src[offset + 0];
        dest[i]->_mp_prec = src[offset + 1];
        dest[i]->_mp_exp = src[offset + 2];
        dest[i]->_mp_d = malloc(NLIMBS * sizeof(mp_limb_t));
        memcpy(dest[i]->_mp_d, &src[offset + 3], NLIMBS * sizeof(mp_limb_t));
        offset += PACKED_LIMBS;
    }
    return dest;
}

void packed_add(void *__in, void *__inout, int *len, MPI_Datatype *datatype)
{
    mp_limb_t *_in = (mp_limb_t *)__in;
    mp_limb_t *_inout = (mp_limb_t *)__inout;

    mpf_t *in = unpack(NULL, _in, *len);
    mpf_t *inout = unpack(NULL, _inout, *len);

    for (int i = 0; i < *len; i++)
        mpf_add(inout[i], in[i], inout[i]);

    pack(_inout, inout, *len);
    free(in);
    free(inout);
}

int main(int argc, char **argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpf_set_default_prec(PREC);

    // Create an MPI Datatype based on the target precision
    MPI_Datatype MPI_MPF;
    MPI_Type_contiguous(PACKED_BYTES, MPI_CHAR, &MPI_MPF);
    MPI_Type_commit(&MPI_MPF);

    // Create an MPI Operation for adding mpfs
    MPI_Op MPI_SUM_MPF;
    MPI_Op_create(packed_add, 1, &MPI_SUM_MPF);

    // Create the number 1
    mpf_t n;
    mpf_init(n);
    mpf_add_ui(n, n, 1);

    // Pack the number and sum over all processes
    mp_limb_t *packed = pack(NULL, &n, 1);
    mp_limb_t *sum_packed = malloc(PACKED_BYTES);
    MPI_Reduce(packed, sum_packed, 1, MPI_MPF, MPI_SUM_MPF, 0, MPI_COMM_WORLD);

    // Unpack the sum and print it
    if (rank == 0) {
        mpf_t sum;
        unpack(&sum, sum_packed, 1);
        mp_exp_t exp;
        char *repr = mpf_get_str(NULL, &exp, 10, 0, sum);
        printf("value: 0.%s * 10^%d\n", repr, exp);
    }

    free(packed);
    free(sum_packed);

    MPI_Finalize();
    return 0;
}
