#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include <math.h>
#include <stdint.h> 
#include "segy.h"
#include<omp.h>
#include "fd.h"
#include <libxml/parser.h>
#include <libxml/tree.h>

void write_text_header(FILE *fp) {
    char text_header[3200];
    memset(text_header, ' ', 3200);

    const char *line1 = "C01 SEG-Y file created by custom C code";
    strncpy(text_header, line1, strlen(line1));

    fwrite(text_header, 1, 3200, fp);
}

void write_binary_header(FILE *fp, short ns, short dt) {
    unsigned char binary_header[400];
    memset(binary_header, 0, 400);

    binary_header[16] = (dt >> 8) & 0xFF;
    binary_header[17] = dt & 0xFF;

    binary_header[20] = (ns >> 8) & 0xFF;
    binary_header[21] = ns & 0xFF;

    binary_header[24] = 0;
    binary_header[25] = 5;

    fwrite(binary_header, 1, 400, fp);
}

short swap_short(short val) {
    return (val << 8) | ((val >> 8) & 0xFF);
}

int swap_int(int val) {
    return ((val >> 24) & 0xff) |
           ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) |
           ((val << 24) & 0xff000000);
}

float swap_float(float val) {
    union {
        float f;
        uint8_t b[4];
    } src, dst;

    src.f = val;
    dst.b[0] = src.b[3];
    dst.b[1] = src.b[2];
    dst.b[2] = src.b[1];
    dst.b[3] = src.b[0];
    return dst.f;
}


void index_shot_3d(int ns,int ns_x,int **table,int disx_shot,int disy_shot,int sourcex_min_grid,int sourcey_min_grid,int dx,int dy,int disx,int disy)
{

	for(int i=0;i<ns;i++)
		
		{
		int tmpx = i%ns_x;

		int tmpy = floor(i/ns_x);

		table[i][0] = i+1;	//shot number
		table[i][1] = sourcex_min_grid*dx + disx_shot*tmpx;          //source position x
		table[i][2] = 0;		// receiver position Xmin
		table[i][3] = disx;		// receiver position Xmax
		table[i][4] = sourcey_min_grid*dy + disy_shot*tmpy;          //source position y
		table[i][5] = 0;		// receiver position Ymin
		table[i][6] = disy;		// receiver position Ymax
		}

}

void    array1d_setPml_3d(float *vp, int nx, int ny, int nz, float dz, float dy, float dx,int pml, int nxpml, int nypml, int nzpml, float *d_dampz,float *d_dampx,float *d_dampy)
{

	float rr=1e-7f;

	for(int iy=0; iy<nypml; iy++)
	for(int ix=0; ix<nxpml; ix++)
	for(int iz=0; iz<nzpml; iz++)
	{
		if(iz<pml){
			d_dampz[iy*nzpml*nxpml+ix*nzpml+iz] = log10(1/rr)*(50.0*vp[iy*nzpml*nxpml+ix*nzpml+iz]/(2.0*pml))*powf(1.0*(pml-iz)/pml,4.0);
		}
		if(iz>nzpml-pml){
			d_dampz[iy*nzpml*nxpml+ix*nzpml+iz] = log10(1/rr)*(50.0*vp[iy*nzpml*nxpml+ix*nzpml+iz]/(2.0*pml))*powf(1.0*(iz-(nzpml-pml))/pml,4.0);
		}	
	
		if(ix<pml){
			d_dampx[iy*nzpml*nxpml+ix*nzpml+iz] = log10(1/rr)*(50.0*vp[iy*nzpml*nxpml+ix*nzpml+iz]/(2.0*pml))*powf(1.0*(pml-ix)/pml,4.0);
		}
		if(ix>nxpml-pml){
			d_dampx[iy*nzpml*nxpml+ix*nzpml+iz] = log10(1/rr)*(50.0*vp[iy*nzpml*nxpml+ix*nzpml+iz]/(2.0*pml))*powf(1.0*(ix-(nxpml-pml))/pml,4.0);
		}		

		if(iy<pml){
			d_dampy[iy*nzpml*nxpml+ix*nzpml+iz] = log10(1/rr)*(50.0*vp[iy*nzpml*nxpml+ix*nzpml+iz]/(2.0*pml))*powf(1.0*(pml-iy)/pml,4.0);
		}
		if(iy>nypml-pml){
			d_dampy[iy*nzpml*nxpml+ix*nzpml+iz] = log10(1/rr)*(50.0*vp[iy*nzpml*nxpml+ix*nzpml+iz]/(2.0*pml))*powf(1.0*(iy-(nypml-pml))/pml,4.0);
		}	

	}

}


void    array1d_setVel_3d(float *v,int ny, int nx, int nz, int pml)
{
        int nzpml = nz+2*pml;
        int nxpml = nx+2*pml;
        int nypml = ny+2*pml;

	int nxnz=nxpml*nzpml;
	
	for(int iy=0;iy<nypml;++iy)
		for(int ix=0;ix<nxpml;++ix)
			for(int iz=0;iz<pml;++iz)
			{
				v[iy*nxnz+ix*nzpml+iz] = v[iy*nxnz+ix*nzpml+pml];
				v[iy*nxnz+ix*nzpml+nzpml-pml+iz] = v[iy*nxnz+ix*nzpml+nzpml-pml-1];
			}

	for(int iy=0;iy<nypml;++iy)
		for(int ix=0;ix<pml;++ix)
			for(int iz=0;iz<nzpml;++iz)
			{
				v[iy*nxnz+ix*nzpml+iz] = v[iy*nxnz+pml*nzpml+iz];
				v[iy*nxnz+(nxpml-pml+ix)*nzpml+iz] = v[iy*nxnz+(nxpml-pml-1)*nzpml+iz];
			}

	for(int iy=0;iy<pml;++iy)
		for(int ix=0;ix<nxpml;++ix)
			for(int iz=0;iz<nzpml;++iz)
			{
				v[iy*nxnz+ix*nzpml+iz] = v[pml*nxnz+ix*nzpml+iz];
				v[(nypml-pml+iy)*nxnz+ix*nzpml+iz] = v[(nypml-pml-1)*nxnz+ix*nzpml+iz];
			}

}


void  ricker1 (int nt,  float f, float dt,float *s)
{
        float pi = 3.14159265358979f;
        float t0 = 1/f;
        int   kt = (int)(t0/dt);

        for(int i=0; i<nt; i++){
           float tt = i*dt-kt*dt;
           float sp = pi*f*tt;
           s[i] = 1.0*exp(-sp*sp)*(1.-2.*sp*sp); 
        }
}

void read_parameters(const char *filename, modelpar *model,int myid) {
    xmlDoc *document = xmlReadFile(filename, NULL, 0);
    if (document == NULL) {
        fprintf(stderr, "Failed to parse XML file: %s\n", filename);
        return;
    }

    xmlNode *root = xmlDocGetRootElement(document);
    xmlNode *current_node = root->children;

    while (current_node) {
        if (current_node->type == XML_ELEMENT_NODE) {
            xmlChar *name = xmlGetProp(current_node, (const xmlChar *)"name");
            xmlChar *value = xmlNodeGetContent(current_node);

            if (strcmp((char *)name, "fm") == 0) {
                model->fm = strtof((const char *)value, NULL);
                if(myid==0)printf("model->fm = %f\n",model->fm);
            }           

            if (strcmp((char *)name, "dy") == 0) {
                model->dy = strtof((const char *)value, NULL);
                if(myid==0)printf("model->dy = %f\n",model->dy);
            }

            if (strcmp((char *)name, "dx") == 0) {
                model->dx = strtof((const char *)value, NULL);
                if(myid==0)printf("model->dx = %f\n",model->dx);
            }
            if (strcmp((char *)name, "dz") == 0) {
                model->dz = strtof((const char *)value, NULL);
                if(myid==0)printf("model->dz = %f\n",model->dz);
            }

            if (strcmp((char *)name, "pml") == 0) {
                model->pml = atoi((const char *)value);
                if(myid==0)printf("model->pml = %d\n",model->pml);
            }

            if (strcmp((char *)name, "ny") == 0) {
                model->ny = atoi((const char *)value);
                if(myid==0)printf("model->ny = %d\n",model->ny);
            }

            if (strcmp((char *)name, "nx") == 0) {
                model->nx = atoi((const char *)value);
                if(myid==0)printf("model->nx = %d\n",model->nx);
            }
            if (strcmp((char *)name, "nz") == 0) {
                model->nz = atoi((const char *)value);
                if(myid==0)printf("model->nz = %d\n",model->nz);
            }       
        
            if (strcmp((char *)name, "dt") == 0) {
                model->dt = strtof((const char *)value, NULL);
                if(myid==0)printf("model->dt = %f\n",model->dt);
            }
            if (strcmp((char *)name, "t") == 0) {
                model->t = strtof((const char *)value, NULL);
                if(myid==0)printf("model->t = %f\n",model->t);
            }

            if (strcmp((char *)name, "disx_shot_grid") == 0) {
                model->disx_shot_grid = atoi((const char *)value);
                if(myid==0)printf("model->disx_shot_grid = %d\n",model->disx_shot_grid);
            }

            if (strcmp((char *)name, "disy_shot_grid") == 0) {
                model->disy_shot_grid = atoi((const char *)value);
                if(myid==0)printf("model->disy_shot_grid = %d\n",model->disy_shot_grid);
            }

            if (strcmp((char *)name, "sourcex_min_grid") == 0) {
                model->sourcex_min_grid = atoi((const char *)value);
                if(myid==0)printf("model->sourcex_min_grid = %d\n",model->sourcex_min_grid);
            }			

            if (strcmp((char *)name, "sourcex_max_grid") == 0) {
                model->sourcex_max_grid = atoi((const char *)value);
                if(myid==0)printf("model->sourcex_max_grid = %d\n",model->sourcex_max_grid);
            }	

            if (strcmp((char *)name, "sourcey_min_grid") == 0) {
                model->sourcey_min_grid = atoi((const char *)value);
                if(myid==0)printf("model->sourcey_min_grid = %d\n",model->sourcey_min_grid);
            }	

            if (strcmp((char *)name, "sourcey_max_grid") == 0) {
                model->sourcey_max_grid = atoi((const char *)value);
                if(myid==0)printf("model->sourcey_max_grid = %d\n",model->sourcey_max_grid);
            }	

            if (strcmp((char *)name, "sz") == 0) {
                model->sz = atoi((const char *)value);
                if(myid==0)printf("model->sz = %d\n",model->sz);
            }	

            if (strcmp((char *)name, "gz") == 0) {
                model->gz = atoi((const char *)value);
                if(myid==0)printf("model->gz = %d\n",model->gz);
            }				

            if (strcmp((char *)name, "scale_y") == 0) {
                model->scale_y = atoi((const char *)value);
                if(myid==0)printf("model->scale_y = %d\n",model->scale_y);
            }	

            if (strcmp((char *)name, "scale_x") == 0) {
                model->scale_x = atoi((const char *)value);
                if(myid==0)printf("model->scale_x = %d\n",model->scale_x);
            }	

            if (strcmp((char *)name, "gpu_num") == 0) {
                model->gpu_num = atoi((const char *)value);
                if(myid==0)printf("model->gpu_num = %d\n",model->gpu_num);
            }	

            if (strcmp((char *)name, "id_of_group") == 0) {
                model->id_of_group = atoi((const char *)value);
                if(myid==0)printf("model->id_of_group = %d\n",model->id_of_group);
            }	

            if (strcmp((char *)name, "fvel") == 0) {
                strncpy(model->fvel, (const char *)value, sizeof(model->fvel) - 1);
                model->fvel[sizeof(model->fvel) - 1] = '\0'; 
                if(myid==0)printf("model->fvel: %s\n", model->fvel);
            }

            if (strcmp((char *)name, "outfile") == 0) {
                strncpy(model->outfile, (const char *)value, sizeof(model->outfile) - 1);
                model->outfile[sizeof(model->outfile) - 1] = '\0'; 
                if(myid==0)printf("model->outfile: %s\n", model->outfile);
            }

            xmlFree(name);
            xmlFree(value);
        }
        current_node = current_node->next;
    }

    xmlFreeDoc(document);
    xmlCleanupParser();
}



int main(int argc,char *argv[]){

	char parfn[1024];
    int i,j,m,nt;

	strcpy(parfn,argv[1]);

	int ns_x,ns_y,ns;
	int gpu_num,id_of_group;

	modelpar model;

	int myid,np;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&np);

    read_parameters(argv[1],&model,myid);

	ns_x = model.sourcex_max_grid - model.sourcex_min_grid + 1;
	ns_y = model.sourcey_max_grid - model.sourcey_min_grid + 1;

	ns = ns_x * ns_y;

	// std::cout<<"ns_x = "<<ns_x<<"\t ns_y = "<<ns_y<<"\t ns = "<<ns<<std::endl;

	nt = (int)(model.t/model.dt)+1;

	int disy = (model.ny-1)*model.dy;
	int disx = (model.nx-1)*model.dx;
	int disz = (model.nz-1)*model.dz;

	int **table = new int*[ns];

	for (int i = 0; i < ns; ++i) {
		table[i] = new int[7];   
	}

	index_shot_3d(ns,ns_x,table,model.disx_shot_grid,model.disy_shot_grid,model.sourcex_min_grid,model.sourcey_min_grid,model.dx,model.dy,disx,disy);

	int nzpml = model.nz + 2*model.pml;
	int nxpml = model.nx + 2*model.pml;
	int nypml = model.ny + 2*model.pml;

	float *vel = NULL;	

	segy1 Th;

	FILE *fp1 = NULL;
	fp1 = fopen(model.fvel,"rb");

    int nxnz = model.nx*model.nz;

	if(myid!=0){
		vel = new float[model.nz*model.nx*model.ny];
		for(i=0;i<model.ny;i++)
		for(j=0;j<model.nx; j++)
		{
			fread(&vel[i*nxnz+j*model.nz],sizeof(float),model.nz,fp1);	
		}

		// for(int i=0;i<model.ny;i++)
		// for(int j=0;j<model.nx;j++)
		// for(int m=0;m<model.nz;m++)	
		// {
		// 	vel[i*nxnz+j*model.nz+m] = 3000;			
		// }		
	}

	fclose(fp1);		

	float *sou = NULL;
	sou = new float[nt];

	ricker1(nt,model.fm,model.dt,sou);

	float *vt = NULL ;
	float *p  = NULL ;
	float *rho  = NULL ;
	float *k  = NULL ;
	float *dampz = NULL;
	float *dampx = NULL;	
	float *dampy = NULL;	

	if(myid!=0){
		vt = new float[nzpml*nxpml*nypml];
		p  = new float[nzpml*nxpml*nypml];
		rho = new float[nzpml*nxpml*nypml];
		k  = new float[nzpml*nxpml*nypml];
		dampz = new float[nzpml*nxpml*nypml];
		dampx = new float[nzpml*nxpml*nypml];
		dampy = new float[nzpml*nxpml*nypml];

		memset(dampz, 0, sizeof(float) * nzpml * nxpml * nypml);		
		memset(dampx, 0, sizeof(float) * nzpml * nxpml * nypml);	
		memset(dampy, 0, sizeof(float) * nzpml * nxpml * nypml);				
	}

	int ytrace = 0;
	int xtrace = 0;

	for(int i=0;i<model.nx;i++){
		if(i%model.scale_x==0){
			xtrace++;
		}
	}
	for(int i=0;i<model.ny;i++){
		if(i%model.scale_y==0){
			ytrace++;
		}
	}

	int ntpoint = nt;	
	float *rec = NULL;

	int nxz;

	if(myid==0) printf("nt = %d\n", nt);

	if(myid==0) printf("all shots = %d\n", ns);
	int ip;
	int send[7],recv[7];
	int nsend,ntask;
	ntask = ns;
	MPI_Barrier(MPI_COMM_WORLD);
	FILE *fp0 = NULL;
	fp0 = fopen(model.outfile,"wb");

	if(myid==0){
		write_text_header(fp0);
		short  ns_segy;
		short  dt_segy;
		ns_segy = nt;
		dt_segy = model.dt*1e6;

		write_binary_header(fp0, ns_segy, dt_segy);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	cudaSetDevice((myid-1)%model.gpu_num);		

	int group_id = MPI_UNDEFINED;			//group id
	int group_rank=-1;						//myid in group
	int group_size=-1;
	int group_num = (np - 1)/model.id_of_group;	// How many groups

	if(myid!=0){
		group_id = (myid - 1)  / model.id_of_group;
	}

	MPI_Comm group_comm;
	MPI_Comm_split(MPI_COMM_WORLD, group_id, myid, &group_comm);

	if (group_id != MPI_UNDEFINED) {
		MPI_Comm_rank(group_comm, &group_rank);
		MPI_Comm_size(group_comm, &group_size);
		std::cout<<"myid = "<<myid<<", group_id = "<<group_id<<", group_rank = "<<group_rank<<" group_size = "<<group_size<<std::endl;		
	}

	if(group_rank==0){

		long long all_trace_nt = xtrace*ytrace*1LL*ntpoint;

		double memory_record = (double)all_trace_nt*4/1024/1024/1024;

		std::cout<<"all_trace*nt = "<<all_trace_nt<<"  Record Memory Allocate = "<<memory_record<<"GB"<<std::endl;

		rec = new float[all_trace_nt];

		memset(rec,0,all_trace_nt*sizeof(float));
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(myid==0)
	{
		nsend = 0;
		for(i=0;i<ntask+group_num;i++)
		{
			MPI_Recv(recv,7,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			ip = status.MPI_SOURCE;
			if(i<ns)
			{
				send[0] = table[i][0];           //shot number
				send[1] = table[i][1];		     //source location x
				send[2] = table[i][2];		     //gx_min
				send[3] = table[i][3];		     //gx_max
				send[4] = table[i][4];		     //source location y
				send[5] = table[i][5];		     //gy_min
				send[6] = table[i][6];		     //gy_max
			}
			else
			{
				// printf("shotnum = %d\n",i);				
				send[0] = 0;
			}
			
			MPI_Send(send,7,MPI_INT,ip,99,MPI_COMM_WORLD);
			nsend = nsend+1;
			if(i<ntask)printf("Doing shot gather forward.Send No.=%d. Shot No.=%d to Processor %d\n",nsend,send[0],ip);
		}
	}
	else
	{
		if(group_rank==0){
			MPI_Send(send,7,MPI_INT,0,0,MPI_COMM_WORLD);
		}
		for(;;)
		{
			if(group_rank==0){
				MPI_Recv(recv,7,MPI_INT,0,99,MPI_COMM_WORLD,&status);
			}

			MPI_Bcast(recv, 7, MPI_INT, 0, group_comm);

			MPI_Barrier(group_comm);

			int shotnum = recv[0];
			int sourceloc_x = recv[1];
			int gx_min = recv[2];
			int gx_max = recv[3];
			int sourceloc_y = recv[4];
			int gy_min = recv[5];
			int gy_max = recv[6];
			if(shotnum ==0)
			{
				printf("World_id %d and InGroup_id %d of Group %d has finished task.\n",myid,group_rank,group_id);	
				break;
			}

			int ngx_min = gx_min/model.dx;
			int ngy_min = gy_min/model.dy;

			int sxnum,synum;
			sxnum = (sourceloc_x-gx_min)/model.dx;
			synum = (sourceloc_y-gy_min)/model.dy;

			// printf("sxnum=%d\n",sxnum);
			// printf("synum=%d\n",synum);

			nxz = nxpml*nzpml;

			for(i=0; i<model.ny; i++)
			  for(j=0; j<model.nx; j++)
			    for(m=0; m<model.nz; m++){
				vt[(i+model.pml)*nxz+(j+model.pml)*nzpml+m+model.pml]=vel[(ngy_min+i)*nxnz+(ngx_min+j)*model.nz+m];
			}
			
			array1d_setVel_3d(vt, model.ny, model.nx, model.nz, model.pml);
			
			array1d_setPml_3d(vt, model.ny, model.nx, model.nz, model.dz, model.dx, model.dy, model.pml, nypml,nxpml,nzpml,dampz,dampx,dampy);


		    for(i=0;i<nypml;i++)
			for(j=0; j<nxpml; j++)
			   for(m=0; m<nzpml; m++){
			   rho[i*nxz+j*nzpml+m] = 2000;
			   k[i*nxz+j*nzpml+m]=vt[i*nxz+j*nzpml+m]*vt[i*nxz+j*nzpml+m]*rho[i*nxz+j*nzpml+m];
			}
			
			int ntr = 0;
			for(int iy = 0; iy < model.ny; iy++)
				for (int ix = 0; ix < model.nx; ix++)
				{
					if(ix%model.scale_x==0&&iy%model.scale_y==0) {ntr++;}
				}


			int ntnum = nt;

			// printf("ntnum=%d\n",ntnum);

			// printf("ytrace=%d,xtrace=%d,ntr=%d\n",ytrace,xtrace,ntr);

		    FD_extrapolation(p,k,rho,dampz,dampx,dampy,rec,nypml,nxpml,nzpml,model.dt,synum,sxnum,sou,model.dx,model.dy,model.dz,nt,model.pml,model.nx,model.ny,xtrace,model.scale_y,model.scale_x,model.sz,model.gz,myid,model.gpu_num,ns_x,ns_y,group_rank,group_id,group_num,model.id_of_group,group_comm);

			MPI_Barrier(group_comm);

			if(group_rank==0){
				long long offsett = 3600 + (240+sizeof(float)*ntnum)*1LL*ntr*(shotnum-1);
				int numm = (offsett - 3600)/(240+4*ntnum)/ntr;

				// printf("pointer offset is %ld, numm is %d\n",offsett,numm);

				fseek(fp0,offsett,SEEK_SET);
				short  nss;
				short  hdt;			  
				nss = ntnum;
				hdt= model.dt*1e6;
				memset(&Th,0,sizeof(segy1));

				for(int ii=0; ii<ytrace; ii++)
					for(int kk=0; kk<xtrace; kk++)
						for(int jj=0; jj<nt; jj++)
						{

							Th.tracl =swap_int(ii*xtrace+kk+1);         // trace sequence number within line //
							Th.fldr  =swap_int(shotnum);           //9-12    // field record number //
							Th.ep    =swap_int(shotnum);  //17-20   // energy source point number //
							Th.cdp   =swap_int(ii*xtrace+kk+1);  //21-24
							Th.sx    = swap_int(sourceloc_x);           //73-76
							Th.sy    = swap_int(sourceloc_y);           //77-80
							Th.gx    = swap_int((int)(kk*model.dx*model.scale_x)+gx_min);
							Th.gy    = swap_int((int)(ii*model.dy*model.scale_y)+gy_min);           //85-88
							Th.sdel  = swap_int((int)(model.sz*model.dz));
							Th.gdel  = swap_int((int)(model.gz*model.dz));								
							Th.offset=swap_int(kk*model.dx*model.scale_x+gx_min - sourceloc_x);
							Th.ns = swap_short((short)nss);
							Th.dt = swap_short((short)hdt);
							if(jj==0){
								fwrite(&Th,240,1,fp0);
							}
							long long index = jj*xtrace*1LL*ytrace+ii*xtrace+kk;
							float val = swap_float(rec[index]);
							fwrite(&val,sizeof(float),1,fp0);						
						}
				
			}
//			printf("process %d finish writing %d\n",myid,shotnum);
			MPI_Barrier(group_comm);

			if(group_rank==0){
				MPI_Send(send,7,MPI_INT,0,myid,MPI_COMM_WORLD);
			}

		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	fclose(fp0);

	MPI_Finalize();

	if(myid!=0){
		delete[] vt;
		delete[] p;

		delete[] dampz;
		delete[] dampx;
		delete[] dampy;

		delete[] rec;

		delete[] rho;
		delete[] k;
		delete[] vel;
	}

	delete[] sou;

	for (int i = 0; i < ns; ++i) {
		delete[] table[i];
	}
	delete[] table;

	return 0;

}










