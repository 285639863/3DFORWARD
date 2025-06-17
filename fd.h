#include "nccl.h"
#include "mpi.h"
#include <iostream>
using namespace std;

struct cudasize{
	dim3 grid;
	dim3 block;
};

void FD_extrapolation(float *p,float *k,float *rho,float *dampz,float *dampx,float *dampy, float *rec, const int nypml, const int nxpml, const int nzpml, const float dt,const int synum,const int sxnum,\
	float *sou, const float dx,const float dy,const float dz,const int nt,const int pml,const int nx,const int ny,const int xtrace,const int scale_y,const int scale_x,const int sz, const int gz, const int myid,const int gpu_num,const int ns_x,const int ns_y,\
	const int myid_in_group, const int group_id, const int group_num, const int group_ids_num, MPI_Comm group_comm);

typedef struct{
    char fvel[1024];
    char outfile[1024];	
	float fm;
    float dy,dx,dz;
	float dt,t;
	int pml,ny,nx,nz;
	int disx_shot_grid,disy_shot_grid,sourcex_min_grid,sourcex_max_grid,sourcey_min_grid,sourcey_max_grid;
	int sz,gz;
	int scale_y,scale_x;
	int gpu_num,id_of_group;
} modelpar;