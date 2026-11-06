#include <stdio.h>
#include <string.h>
#include <time.h>
#include "blst.h"

static const byte DST[] = "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    const int sizes[] = {4, 7, 13, 31, 61, 103};
    byte msg[32];
    memset(msg, 7, sizeof msg);
    printf("{\n");
    for (int si = 0; si < 6; si++) {
        int k = sizes[si];
        int q = 2 * ((k - 1) / 3) + 1;
        blst_scalar sk[128];
        blst_p1 pk[128];
        blst_p2 sig[128];
        blst_p2 h;
        double t0 = now();
        for (int i = 0; i < q; i++) {
            byte ikm[32];
            memset(ikm, i + 1, sizeof ikm);
            blst_keygen(&sk[i], ikm, sizeof ikm, NULL, 0);
            blst_sk_to_pk_in_g1(&pk[i], &sk[i]);
        }
        double tk = (now() - t0) / q;
        t0 = now();
        for (int i = 0; i < q; i++) {
            blst_hash_to_g2(&h, msg, sizeof msg, DST, sizeof DST - 1, NULL, 0);
            blst_sign_pk_in_g1(&sig[i], &h, &sk[i]);
        }
        double ts = (now() - t0) / q;
        t0 = now();
        blst_p2 asig = sig[0];
        blst_p1 apk = pk[0];
        for (int i = 1; i < q; i++) {
            blst_p2_add_or_double(&asig, &asig, &sig[i]);
            blst_p1_add_or_double(&apk, &apk, &pk[i]);
        }
        double ta = now() - t0;
        blst_p1_affine apk_a;
        blst_p2_affine asig_a;
        blst_p1_to_affine(&apk_a, &apk);
        blst_p2_to_affine(&asig_a, &asig);
        t0 = now();
        int reps = 20;
        BLST_ERROR e = BLST_SUCCESS;
        for (int r = 0; r < reps; r++)
            e = blst_core_verify_pk_in_g1(&apk_a, &asig_a, 1, msg, sizeof msg, DST, sizeof DST - 1, NULL, 0);
        double tv = (now() - t0) / reps;
        printf("  \"K=%d\": {\"quorum\": %d, \"keygen_ms\": %.3f, \"sign_ms\": %.3f, \"aggregate_ms\": %.3f, \"verify_ms\": %.3f, \"valid\": %s}%s\n",
               k, q, tk * 1e3, ts * 1e3, ta * 1e3, tv * 1e3, e == BLST_SUCCESS ? "true" : "false", si < 5 ? "," : "");
    }
    printf("}\n");
    return 0;
}
