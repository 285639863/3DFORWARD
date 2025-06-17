#include<stdio.h>
#include<stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "fd.h"

#define BLOCKDIMX 32
#define BLOCKDIMY 32
#define nop 4


__global__ void testcalculate_p(float *snap,const int nDim,const int nypml,const int nxpml, const int nzpml,const float dt,const float dx,const float dy,const float dz,float *coeff)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
	int iy;
	const long index = ix*nzpml+iz;
	__syncthreads();
//	const int loop = nzpml;
	const int nxnz = nzpml*nxpml;
	for(int iter=nop;iter<=nypml-nop;iter++)
	{
		iy = iter;

		long long index_all = 9*1LL*nDim+index+iy*nxnz;

		float damp1 = 1 - dt*snap[index_all]/2;
		float damp2 = 1 + dt*snap[index_all]/2;

		float damp3 = 1 - dt*snap[index_all+nDim]/2;
		float damp4 = 1 + dt*snap[index_all+nDim]/2;		

		float damp5 = 1 - dt*snap[index_all+2*nDim]/2;
		float damp6 = 1 + dt*snap[index_all+2*nDim]/2;		

		float tmp_vy = 0;
		float tmp_vx = 0;
		float tmp_vz = 0;

//#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vy += coeff[i]*(snap[4*nDim+(iy+i)*nxnz+ix*nzpml+iz]-snap[4*nDim+(iy-i+1)*nxnz+ix*nzpml+iz]);
			tmp_vx += coeff[i]*(snap[5*nDim+iy*nxnz+(ix+i)*nzpml+iz]-snap[5*nDim+iy*nxnz+(ix-i+1)*nzpml+iz]);
			tmp_vz += coeff[i]*(snap[6*nDim+iy*nxnz+ix*nzpml+(iz+i)]-snap[6*nDim+iy*nxnz+ix*nzpml+(iz-i+1)]);

		}
		
		snap[index+iy*nxnz] = (damp5*snap[index+iy*nxnz]-snap[7*nDim+index+iy*nxnz]*(dt/dy)*tmp_vy)/damp6;
		snap[nDim+index+iy*nxnz] = (damp3*snap[nDim+index+iy*nxnz]-snap[7*nDim+index+iy*nxnz]*(dt/dx)*tmp_vx)/damp4;	
		snap[2*nDim+index+iy*nxnz] = (damp1*snap[2*nDim+index+iy*nxnz]-snap[7*nDim+index+iy*nxnz]*(dt/dz)*tmp_vz)/damp2;//snap[6*nDim+iy*nxnz+ix*nzpml+iz];
		snap[3*nDim+index+iy*nxnz] = snap[index+iy*nxnz]+snap[nDim+index+iy*nxnz]+snap[2*nDim+index+iy*nxnz];

	}

}

__global__ void testcalculate_v_xyz(float *snap,const int nDim,const int nypml,const int nxpml, const int nzpml,const float dt,const float dz,const float dx,const float dy,float *coeff)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
	int iy;
	const long long index = ix*nzpml+iz;
	__syncthreads();
	const int nxnz = nzpml*nxpml;
	for(int iter=nop;iter<=nypml-nop;iter++)
	{
		iy = iter;

		long long index_all = 9*1LL*nDim+index+iy*nxnz;

		float damp1 = 1 - dt*snap[index_all]/2;
		float damp2 = 1 + dt*snap[index_all]/2;

		float damp3 = 1 - dt*snap[index_all+nDim]/2;
		float damp4 = 1 + dt*snap[index_all+nDim]/2;		

		float damp5 = 1 - dt*snap[index_all+2*nDim]/2;
		float damp6 = 1 + dt*snap[index_all+2*nDim]/2;		

		float tmp_p_z = 0;
		float tmp_p_x = 0;
		float tmp_p_y = 0;

		long long index_finite = 3*1LL*nDim;

		for(int i=1;i<=nop;i++)
		{
			tmp_p_z += coeff[i]*(snap[index_finite+iy*nxnz+ix*nzpml+(iz+i-1)]-snap[index_finite+iy*nxnz+ix*nzpml+(iz-i)]);

			tmp_p_x += coeff[i]*(snap[index_finite+iy*nxnz+(ix+i-1)*nzpml+iz]-snap[index_finite+iy*nxnz+(ix-i)*nzpml+iz]);		

			tmp_p_y += coeff[i]*(snap[index_finite+(iy+i-1)*nxnz+ix*nzpml+iz]-snap[index_finite+(iy-i)*nxnz+ix*nzpml+iz]);				

		}
		
		snap[index_finite+1*nDim+index+iy*nxnz] = (damp5*snap[index_finite+1*nDim+index+iy*nxnz]-(1.0f/snap[index_finite+5*nDim+iy*nxnz])*(dt/dy)*tmp_p_y)/damp6;

		snap[index_finite+2*nDim+index+iy*nxnz] = (damp3*snap[index_finite+2*nDim+index+iy*nxnz]-(1.0f/snap[index_finite+5*nDim+iy*nxnz])*(dt/dx)*tmp_p_x)/damp4;

		snap[index_finite+3*nDim+index+iy*nxnz] = (damp1*snap[index_finite+3*nDim+index+iy*nxnz]-(1.0f/snap[index_finite+5*nDim+iy*nxnz])*(dt/dz)*tmp_p_z)/damp2;
	}
}

__global__ void applysource(float *snap,const int nDim,float *sou,const int synum,const int sxnum,const int nzpml,const int nxz, const float dt,const float dx,const int it,const int pml,const int sz)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>0)return;

	snap[3*nDim+(synum)*nxz+(sxnum+pml)*nzpml+pml+sz] +=  sou[it];	

}

__global__ void GPUseisrecord_NCCL_Onetime(float *snap,float *receiver,const int nDim, const int nx, const int ny, const int scale_y,const int scale_x, int gz, const int pml,const int nzpml,const int nxpml, \
	int region_index,int ny_afore,int record_ny,int one_swap_layer,int num_region)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	const int iy = blockIdx.y*blockDim.y+threadIdx.y;

	if(iy>=record_ny||ix>=nx)return;

	int ixiy_in_all;

	long index = (iy)*nzpml*nxpml + (ix+pml)*nzpml + (pml+gz);

	int nx_trace = nx / scale_x;

	int ix_trace = ix / scale_x;
	int iy_trace;

	if(region_index==0){

		if(iy<pml||iy>=record_ny-one_swap_layer)return;		

		if( (ix%scale_x==0) && ((iy-pml)%scale_y==0) ){
			iy_trace = (iy - pml) / scale_y;			
			ixiy_in_all = iy_trace*nx_trace + ix_trace;

			receiver[ixiy_in_all]  = snap[3*nDim + index];					
		}
	
	}
	else if(region_index==num_region-1){

		if(iy<one_swap_layer||(iy + region_index * ny_afore - pml - one_swap_layer)>=ny)return;			

		if( (ix%scale_x==0) && ((iy + region_index * ny_afore - pml - one_swap_layer)%scale_y==0) ){
			iy_trace = (iy + region_index * ny_afore - pml - one_swap_layer) / scale_y;			
			ixiy_in_all = iy_trace*nx_trace + ix_trace;

			receiver[ixiy_in_all]  = snap[3*nDim + index];							
		}

	}
	else{

		if(iy<one_swap_layer||iy>=record_ny-one_swap_layer)return;		

		if( (ix%scale_x==0) && ((iy + region_index * ny_afore - pml - one_swap_layer)%scale_y==0) ){
			iy_trace = (iy + region_index * ny_afore - pml - one_swap_layer) / scale_y;			
			ixiy_in_all = iy_trace*nx_trace + ix_trace;

			receiver[ixiy_in_all]  = snap[3*nDim + index];					
		}

	}

}


//FD_extrapolation
void FD_extrapolation(float *p,float *k,float *rho,float *dampz,float *dampx,float *dampy, float *rec, const int nypml, const int nxpml, const int nzpml, const float dt,const int synum,const int sxnum,\
	float *sou, const float dx,const float dy,const float dz,const int nt,const int pml,const int nx,const int ny,const int xtrace,const int scale_y,const int scale_x,const int sz, const int gz, const int myid,const int gpu_num,const int ns_x,const int ns_y,\
	const int myid_in_group, const int group_id, const int group_num, const int group_ids_num, MPI_Comm group_comm)
{

	const int nxz = nzpml*nxpml;

	clock_t t0,t1;

	float C[5];
          C[0] = 0;
          C[1] = 1225.0/1024.0;
          C[2] = -245.0/3072.0;
          C[3] = 49.0/5120.0;
          C[4] = -5.0/7168.0;

	float *coeff;
	cudaMalloc((void**)&coeff,5*sizeof(float));
	cudaMemcpy(coeff,C,5*sizeof(float),cudaMemcpyHostToDevice);

// domain decomposition
	float *source;
	cudaMalloc((void**)&source,nt*sizeof(float));
	if(cudaMemcpy(source,sou,nt*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)printf("please check source\n");

	int device_count = group_ids_num;
	int num_region = group_ids_num;

	ncclComm_t comms;  // 

	ncclUniqueId id;             // 

	if (myid_in_group == 0) {
		ncclGetUniqueId(&id);  // 
	}

	MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, group_comm);

	ncclCommInitRank(&comms, device_count, id, myid_in_group);

	cudaStream_t stream;

	cudaStreamCreate(&stream);

	float *snap_subdomain;

	int ny_afore = ceil((float)nypml/num_region);

	int one_swap_layer = nop;

	int ny_last = nypml - ny_afore * (num_region - 1);

	int region_afore = ny_afore + one_swap_layer;

	int region_last = ny_last + one_swap_layer;

	int region_mid = ny_afore + 2*one_swap_layer;

	int index_first_address_in_cpu_ny =  ((myid_in_group==0) ? ny_afore * myid_in_group : ny_afore * myid_in_group - one_swap_layer);

	// std::cout<<"ny_afore = "<<ny_afore<<" ny_last = "<<ny_last<<std::endl;

	float *swap_n_p,*swap_n_vy,*swap_n_vx,*swap_n_vz;

	int all_swap_layer = ( (num_region==2) ? (2*one_swap_layer) : (2*one_swap_layer + (num_region - 2)*2*one_swap_layer) );		

	cudaMalloc((void**)&swap_n_p,all_swap_layer*nxz*sizeof(float));
	cudaMalloc((void**)&swap_n_vy,all_swap_layer*nxz*sizeof(float));
	cudaMalloc((void**)&swap_n_vx,all_swap_layer*nxz*sizeof(float));
	cudaMalloc((void**)&swap_n_vz,all_swap_layer*nxz*sizeof(float));

	int trace = 0;

	for(int iy = 0; iy < ny; iy++)
		for (int ix = 0; ix < nx; ix++)
		{
			if(ix%scale_x==0&&iy%scale_y==0) {trace++;}
		}

	float *n_receiver;

	cudaMalloc((void**)&n_receiver,trace*sizeof(float));

	int region_ny;

	if(myid_in_group==0){
		region_ny = region_afore;
	}
	else if(myid_in_group==num_region-1){
		region_ny = region_last;
	}
	else{
		region_ny = region_mid;
	}

	int nDim = region_ny*nxz;

	// printf("myid_in_group is %d, the region is %d.\n", myid_in_group,region_ny);
	cudaMalloc((void**)&snap_subdomain,12*region_ny*nxz*sizeof(float));

	cudaMemcpy(&snap_subdomain[7*region_ny*nxz],&k[(index_first_address_in_cpu_ny)*nxz],region_ny*nxz*sizeof(float),cudaMemcpyHostToDevice);			
	cudaMemcpy(&snap_subdomain[8*region_ny*nxz],&rho[(index_first_address_in_cpu_ny)*nxz],region_ny*nxz*sizeof(float),cudaMemcpyHostToDevice);	
	cudaMemcpy(&snap_subdomain[9*region_ny*nxz],&dampz[(index_first_address_in_cpu_ny)*nxz],region_ny*nxz*sizeof(float),cudaMemcpyHostToDevice);			
	cudaMemcpy(&snap_subdomain[10*region_ny*nxz],&dampx[(index_first_address_in_cpu_ny)*nxz],region_ny*nxz*sizeof(float),cudaMemcpyHostToDevice);			
	cudaMemcpy(&snap_subdomain[11*region_ny*nxz],&dampy[(index_first_address_in_cpu_ny)*nxz],region_ny*nxz*sizeof(float),cudaMemcpyHostToDevice);			

	struct cudasize NCCL_Grid_Block;
	struct cudasize NCCL_Record;

	NCCL_Grid_Block.grid.x=(nzpml+BLOCKDIMX-1)/BLOCKDIMX;
	NCCL_Grid_Block.grid.y=(nxpml+BLOCKDIMY-1)/BLOCKDIMY;
	NCCL_Grid_Block.block.x=BLOCKDIMX;
	NCCL_Grid_Block.block.y=BLOCKDIMY;

	int record_ny = region_ny;

	NCCL_Record.grid.x=(nx+BLOCKDIMX-1)/BLOCKDIMX;
	NCCL_Record.grid.y=(record_ny+BLOCKDIMX-1)/BLOCKDIMX;		
	NCCL_Record.block.x=BLOCKDIMX;
	NCCL_Record.block.y=BLOCKDIMX;

	int source_region = (synum + pml)/ ny_afore;

	int source_posion_in_region_grid;

	if(source_region==0){
		source_posion_in_region_grid =  (synum + pml) % ny_afore;
	}
	else{
		source_posion_in_region_grid =  (synum + pml) % ny_afore + one_swap_layer;
	}

	// std::cout<<"source_region = "<<source_region<<" source_posion_in_region_grid = "<<source_posion_in_region_grid<<std::endl;

	t0 = clock();

	for(int it=0;it<nt;it++)
	{
		// if(myid_in_group==0&&it%100==0){
		// 	std::cout<<"sxnum = "<<sxnum<<"\t synum = "<<synum<<"\t it = "<<it<<std::endl;
		// }

		cudaMemset(swap_n_vy,0,all_swap_layer*nxz*sizeof(float));
		cudaMemset(swap_n_vx,0,all_swap_layer*nxz*sizeof(float));
		cudaMemset(swap_n_vz,0,all_swap_layer*nxz*sizeof(float));
		cudaMemset(swap_n_p,0,all_swap_layer*nxz*sizeof(float));								
		cudaMemset(n_receiver,0,trace*sizeof(float));

		if(myid_in_group==source_region){
			applysource<<<1,1>>>(snap_subdomain,nDim,source,source_posion_in_region_grid,sxnum,nzpml,nxz,dt,dx,it,pml,sz);
		}

		MPI_Barrier(group_comm);


		if(myid_in_group==0){			//copy right edge
			cudaMemcpy(&swap_n_p[myid_in_group*(one_swap_layer)*nxz],&snap_subdomain[3*region_ny*nxz+(region_afore-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
							
		}
		else if(myid_in_group==(num_region-1)){								//copy left edge
			cudaMemcpy(&swap_n_p[(all_swap_layer-one_swap_layer)*nxz],&snap_subdomain[3*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);						
		}
		else{
			//left edge in middle region
			cudaMemcpy(&swap_n_p[(2*myid_in_group-1)*(one_swap_layer)*nxz],&snap_subdomain[3*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);

			//right edge in middle region
			cudaMemcpy(&swap_n_p[(2*myid_in_group)*(one_swap_layer)*nxz],&snap_subdomain[3*region_ny*nxz+(region_mid-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);								
		}

		MPI_Barrier(group_comm);

		ncclAllReduce(swap_n_p, swap_n_p, all_swap_layer*nxz, ncclFloat, ncclSum, comms, stream);

		MPI_Barrier(group_comm);

		if(myid_in_group==0){			//copy right edge
			cudaMemcpy(&snap_subdomain[3*region_ny*nxz+(region_afore-one_swap_layer)*nxz],&swap_n_p[(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
		}
		else if(myid_in_group==(num_region-1)){								//copy left edge
			cudaMemcpy(&snap_subdomain[3*region_ny*nxz],&swap_n_p[(all_swap_layer-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);							
		}
		else{
			//left edge
			cudaMemcpy(&snap_subdomain[3*region_ny*nxz],&swap_n_p[(2*myid_in_group-2)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);

			//right edge
			cudaMemcpy(&snap_subdomain[3*region_ny*nxz+(region_mid-one_swap_layer)*nxz],&swap_n_p[(2*myid_in_group+1)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);								
		}

		MPI_Barrier(group_comm);

		testcalculate_v_xyz<<<NCCL_Grid_Block.grid,NCCL_Grid_Block.block>>>(snap_subdomain,nDim,region_ny,nxpml,nzpml,dt,dz,dx,dy,coeff);

		MPI_Barrier(group_comm);

		if(myid_in_group==0){			//copy right edge
			cudaMemcpy(&swap_n_vy[myid_in_group*(one_swap_layer)*nxz],&snap_subdomain[4*region_ny*nxz+(region_afore-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&swap_n_vx[myid_in_group*(one_swap_layer)*nxz],&snap_subdomain[5*region_ny*nxz+(region_afore-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		
			cudaMemcpy(&swap_n_vz[myid_in_group*(one_swap_layer)*nxz],&snap_subdomain[6*region_ny*nxz+(region_afore-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);								
		}
		else if(myid_in_group==(num_region-1)){								//copy left edge
			cudaMemcpy(&swap_n_vy[(all_swap_layer-one_swap_layer)*nxz],&snap_subdomain[4*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&swap_n_vx[(all_swap_layer-one_swap_layer)*nxz],&snap_subdomain[5*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);	
			cudaMemcpy(&swap_n_vz[(all_swap_layer-one_swap_layer)*nxz],&snap_subdomain[6*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);								
		}
		else{
			//left edge in middle region
			cudaMemcpy(&swap_n_vy[(2*myid_in_group-1)*(one_swap_layer)*nxz],&snap_subdomain[4*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&swap_n_vx[(2*myid_in_group-1)*(one_swap_layer)*nxz],&snap_subdomain[5*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		
			cudaMemcpy(&swap_n_vz[(2*myid_in_group-1)*(one_swap_layer)*nxz],&snap_subdomain[6*region_ny*nxz+(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		

			//right edge in middle region
			cudaMemcpy(&swap_n_vy[(2*myid_in_group)*(one_swap_layer)*nxz],&snap_subdomain[4*region_ny*nxz+(region_mid-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&swap_n_vx[(2*myid_in_group)*(one_swap_layer)*nxz],&snap_subdomain[5*region_ny*nxz+(region_mid-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		
			cudaMemcpy(&swap_n_vz[(2*myid_in_group)*(one_swap_layer)*nxz],&snap_subdomain[6*region_ny*nxz+(region_mid-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);									
		}

		MPI_Barrier(group_comm);

		ncclAllReduce(swap_n_vy, swap_n_vy, all_swap_layer*nxz, ncclFloat, ncclSum, comms, stream);
		ncclAllReduce(swap_n_vx, swap_n_vx, all_swap_layer*nxz, ncclFloat, ncclSum, comms, stream);
		ncclAllReduce(swap_n_vz, swap_n_vz, all_swap_layer*nxz, ncclFloat, ncclSum, comms, stream);

		MPI_Barrier(group_comm);

		if(myid_in_group==0){			//copy right edge
			cudaMemcpy(&snap_subdomain[4*region_ny*nxz+(region_afore-one_swap_layer)*nxz],&swap_n_vy[(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&snap_subdomain[5*region_ny*nxz+(region_afore-one_swap_layer)*nxz],&swap_n_vx[(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);	
			cudaMemcpy(&snap_subdomain[6*region_ny*nxz+(region_afore-one_swap_layer)*nxz],&swap_n_vz[(one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);								
		}
		else if(myid_in_group==(num_region-1)){								//copy left edge
			cudaMemcpy(&snap_subdomain[4*region_ny*nxz],&swap_n_vy[(all_swap_layer-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&snap_subdomain[5*region_ny*nxz],&swap_n_vx[(all_swap_layer-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		
			cudaMemcpy(&snap_subdomain[6*region_ny*nxz],&swap_n_vz[(all_swap_layer-2*one_swap_layer)*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);								
		}
		else{
			//left edge
			cudaMemcpy(&snap_subdomain[4*region_ny*nxz],&swap_n_vy[(2*myid_in_group-2)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&snap_subdomain[5*region_ny*nxz],&swap_n_vx[(2*myid_in_group-2)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		
			cudaMemcpy(&snap_subdomain[6*region_ny*nxz],&swap_n_vz[(2*myid_in_group-2)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);	

			//right edge
			cudaMemcpy(&snap_subdomain[4*region_ny*nxz+(region_mid-one_swap_layer)*nxz],&swap_n_vy[(2*myid_in_group+1)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);
			cudaMemcpy(&snap_subdomain[5*region_ny*nxz+(region_mid-one_swap_layer)*nxz],&swap_n_vx[(2*myid_in_group+1)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);		
			cudaMemcpy(&snap_subdomain[6*region_ny*nxz+(region_mid-one_swap_layer)*nxz],&swap_n_vz[(2*myid_in_group+1)*one_swap_layer*nxz],(one_swap_layer)*nxz*sizeof(float),cudaMemcpyDeviceToDevice);									
		}

		MPI_Barrier(group_comm);

		testcalculate_p<<<NCCL_Grid_Block.grid,NCCL_Grid_Block.block>>>(snap_subdomain,nDim,region_ny,nxpml,nzpml,dt,dx,dy,dz,coeff);

		MPI_Barrier(group_comm);

		GPUseisrecord_NCCL_Onetime<<<NCCL_Record.grid,NCCL_Record.block>>>(snap_subdomain,n_receiver,nDim,nx,ny,scale_y,scale_x,gz,pml,nzpml,nxpml,myid_in_group,ny_afore,record_ny,one_swap_layer,num_region);

		MPI_Barrier(group_comm);

		ncclAllReduce(n_receiver, n_receiver, trace, ncclFloat, ncclSum, comms, stream);

		MPI_Barrier(group_comm);		

		if(myid_in_group==0){
			long long index = it*1LL*trace;
			cudaMemcpy(&rec[index],n_receiver,trace*sizeof(float),cudaMemcpyDeviceToHost);
		}

		MPI_Barrier(group_comm);

	}


	t1 = clock();

	printf("time=%.3f(s)\n",((float)(t1-t0))/CLOCKS_PER_SEC);

	ncclCommDestroy(comms);
	cudaStreamDestroy(stream);

	cudaFree(snap_subdomain);	
	cudaFree(n_receiver);

	cudaFree(swap_n_p);
	cudaFree(swap_n_vy);
	cudaFree(swap_n_vx);
	cudaFree(swap_n_vz);

}









