#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>
#include <mpi.h>

#define PREC 100

#define NLIMBS(n)                                                       \
    ((mp_size_t) ((__GMP_MAX (53, n) + 2 * GMP_NUMB_BITS - 1) / GMP_NUMB_BITS))
#define MPF_D_SIZE(prec) (((prec) + 1) * sizeof(mp_limb_t))
#define MPF_PACKED_LIMBS(prec) ((prec) + 4);
#define MPF_PACKED_BYTES ((NLIMBS(mpf_get_default_prec()) + 3) * sizeof(mp_limb_t))

mp_limb_t *mpf_pack(mp_limb_t *dest, mpf_t *src, int n)
{
    if (dest == NULL) {
        size_t size = 0;
        for (int i = 0; i < n; i++)
            size += src[i]->_mp_prec + 4;
        dest = (mp_limb_t *)malloc(size * sizeof(mp_limb_t));
    }
    size_t offset = 0;
    for (int i = 0; i < n; i++) {
        int prec = dest[offset] = src[i]->_mp_prec;
        dest[offset + 1] = src[i]->_mp_size;
        dest[offset + 2] = src[i]->_mp_exp;
        memcpy(&dest[offset + 3], src[i]->_mp_d, MPF_D_SIZE(prec));
        offset += MPF_PACKED_LIMBS(prec);
    }
    return dest;
}

mpf_t *mpf_unpack(mpf_t *dest, mp_limb_t *src, int n)
{
    if (dest == NULL) {
        dest = (mpf_t *)malloc(sizeof(mpf_t) * n);
    }
    else {
        for (int i = 0; i < n; i++)
            if (dest[i]->_mp_d)
                free(dest[i]->_mp_d);
    }
    size_t offset = 0;
    for (int i = 0; i < n; i++) {
        int prec = dest[i]->_mp_prec = src[offset];
        dest[i]->_mp_size = src[offset + 1];
        dest[i]->_mp_exp = src[offset + 2];
        dest[i]->_mp_d = malloc((prec + 1) * sizeof(mp_limb_t));
        memcpy(dest[i]->_mp_d, &src[offset + 3], MPF_D_SIZE(prec));
        offset += MPF_PACKED_LIMBS(prec);
    }
    return dest;
}

void mpf_packed_add(void *_in, void *_inout, int *len, MPI_Datatype *datatype)
{
    mpf_t *in = mpf_unpack(NULL, (mp_limb_t *)_in, *len);
    mpf_t *inout = mpf_unpack(NULL, (mp_limb_t *)_inout, *len);

    for (int i = 0; i < *len; i++)
        mpf_add(inout[i], in[i], inout[i]);

    mpf_pack((mp_limb_t *)_inout, inout, *len);

    for (int i = 0; i < *len; i++) {
        mpf_clear(in[i]);
        mpf_clear(inout[i]);
    }
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
    MPI_Type_contiguous(MPF_PACKED_BYTES, MPI_CHAR, &MPI_MPF);
    MPI_Type_commit(&MPI_MPF);

    // Create an MPI Operation for adding mpfs
    MPI_Op MPI_SUM_MPF;
    MPI_Op_create(mpf_packed_add, 1, &MPI_SUM_MPF);

    // Create the number 1
    mpf_t n;
    mpf_init(n);
    mpf_add_ui(n, n, 1);

    // Pack the number and sum over all processes
    mp_limb_t *packed = mpf_pack(NULL, &n, 1);
    mp_limb_t *sum_packed = malloc(MPF_PACKED_BYTES);
    MPI_Reduce(packed, sum_packed, 1, MPI_MPF, MPI_SUM_MPF, 0, MPI_COMM_WORLD);

    // Unpack the sum and print it
    if (rank == 0) {
        mpf_t sum;
        mpf_unpack(&sum, sum_packed, 1);
        mp_exp_t exp;
        char *repr = mpf_get_str(NULL, &exp, 10, 0, sum);
        printf("value: 0.%s * 10^%d\n", repr, exp);
        mpf_clear(sum);
    }

    free(packed);
    free(sum_packed);
    mpf_clear(n);

    MPI_Finalize();
    return 0;
}
