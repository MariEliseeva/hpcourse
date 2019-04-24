__kernel void convolution(__global double * a, __global double * b,
                                   __global double * c, int N, int M){
    int idx = get_global_id(0);
    if (idx < N * N) {
        HM = (M - 1) / 2
        i = id / N
        j = id % N
        for (int k = -HM; k < HM; k++){
            for (int l = -HM; l < HM; l++) {
                if (i + k >= 0 && j + l >= 0 && i + k < N && j + l < N) {
                    c[i * N + j] += a[(i + k) * N +  j + l] * b[(k + HM) * M + l + HM]
                }
            }
        }
    }
}