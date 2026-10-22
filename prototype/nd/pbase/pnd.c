// ParMETIS_V3_NodeND driver. usage: mpirun -n P pnd graph.csr out.order
// graph.csr: int32 n, int32 nnz, int32 xadj[n+1], int32 adjncy[nnz] (0-based, symmetric, no diagonal)
// out.order: int32 order[n] (order[v] = elimination position of v). prints "PNDTIME seconds" on rank 0.
#include <mpi.h>
#include <parmetis.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int P, r; MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &P); MPI_Comm_rank(comm, &r);
  FILE *f = fopen(argv[1], "rb"); if (!f) { fprintf(stderr, "open\n"); MPI_Abort(comm, 1); }
  int32_t hdr[2]; if (fread(hdr, 4, 2, f) != 2) MPI_Abort(comm, 1);
  idx_t n = hdr[0]; long nnz = hdr[1];
  idx_t *vtxdist = malloc((P + 1) * sizeof(idx_t));
  for (int i = 0; i <= P; i++) vtxdist[i] = (idx_t)(((long)n * i) / P);
  idx_t lo = vtxdist[r], hi = vtxdist[r + 1], nl = hi - lo;
  idx_t *xadj = malloc((nl + 1) * sizeof(idx_t));
  fseek(f, 8 + 4L * lo, SEEK_SET);
  if (fread(xadj, 4, nl + 1, f) != (size_t)(nl + 1)) MPI_Abort(comm, 1);
  long e0 = xadj[0], e1 = xadj[nl], m = e1 - e0;
  idx_t *adj = malloc((m > 0 ? m : 1) * sizeof(idx_t));
  fseek(f, 8 + 4L * (n + 1) + 4L * e0, SEEK_SET);
  if (fread(adj, 4, m, f) != (size_t)m) MPI_Abort(comm, 1);
  fclose(f);
  for (idx_t i = 0; i <= nl; i++) xadj[i] -= (idx_t)e0;
  idx_t numflag = 0, options[3] = {0, 0, 0};
  idx_t *order = malloc((nl > 0 ? nl : 1) * sizeof(idx_t));
  idx_t *sizes = malloc(2 * P * sizeof(idx_t));
  MPI_Barrier(comm); double t0 = MPI_Wtime();
  int rc = ParMETIS_V3_NodeND(vtxdist, xadj, adj, &numflag, options, order, sizes, &comm);
  MPI_Barrier(comm); double t1 = MPI_Wtime();
  if (rc != METIS_OK) { fprintf(stderr, "rc=%d\n", rc); MPI_Abort(comm, 2); }
  int *cnt = NULL, *dsp = NULL; idx_t *all = NULL;
  if (r == 0) { cnt = malloc(P * sizeof(int)); dsp = malloc(P * sizeof(int)); all = malloc((size_t)n * sizeof(idx_t));
    for (int i = 0; i < P; i++) { cnt[i] = vtxdist[i + 1] - vtxdist[i]; dsp[i] = vtxdist[i]; } }
  MPI_Gatherv(order, nl, MPI_INT, all, cnt, dsp, MPI_INT, 0, comm);
  if (r == 0) { printf("PNDTIME %.6f\n", t1 - t0); FILE *o = fopen(argv[2], "wb"); fwrite(all, 4, n, o); fclose(o); }
  MPI_Finalize(); return 0;
}
