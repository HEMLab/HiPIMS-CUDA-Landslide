// ======================================================================================
// Name                :    Drainage model (first version)
// Description         :    This drainage model is developed by coupling TPA and 2D SWEs
// ======================================================================================
// Version             :    1.00
// Authors             :    Qian Li                                                   
//                           PhD candidate in Newcastle University
// Create Time         :    2018/4/30
// Update Time         :    
// ======================================================================================
// Copyright @	Qian Li . All rights reserved.
// ======================================================================================


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "drainage_class.h"
#include "drainage_functions.h"
#include "drainage_interface.h"
#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <map>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include "mapped_field.h"
#include "cuda_mapped_field.h"




namespace GC {

	//--------global functions list------------------------------------------------------------------------------------------------------////////////////////////////
	__global__ void surfDrainQKernal_calculation( int num_Junc,
		 double g,
		 double pi,
		 double dT,
		 double dx,
		 double *Qe,
		double *Qdrainage,
		double *juncMaxDept,
		double *juncR,
		double *juncArea,
		double *hj,
		double *zbJunc,
		int *index,
		int *surfExGrid_device,
		int junc_expanValue,
		double *zbSurf,
		double *hs,
		double *QSurf);
		__global__ void surfH_limilatorKernal(unsigned int num_Girds, double *h_surf_device);

		__global__ void junction_IntegrateQKernel(int junc_number, double dt, double *h_j_device,double *hn_j_device, 
										 double* Qexchange_device);

	  __device__ void QeLimitor(double &QeD, double dx, double hs, double juncArea, double hj, double dT, double maxDept);

      __device__ void  QexchangeCalculation(double hs, double hj, double juncArea, double dx, double Ci, double pi, double g,
    	                                  double kesi, double juncMaxDept, double &tempSingleQ, int juncSurfSize, double dT);

     __device__ void  Q_maximumStorage_limit(double &Qe, double &tempSingleQ, double juncMaxDept, 
    	                                    double hj, double juncArea, double dt, double kesi,
    	                                    int junc_expanValue);


	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	void surfDrainQ_calculation(double dT, Input_information inputKeyPtr, Junc_variables &juncVariables,
		Const_variables constVariables, cuFvMappedField<Scalar, on_cell>& z, cuFvMappedField<Scalar, on_cell>& h, cuFvMappedField<Scalar, on_cell>& Q_surface,
		int blocksPerGrid, int threadsPerBlock) {
		auto mesh = h.mesh;
		auto cell_volumes = mesh->cell_volumes.host_ptr();
		double dx_surface = sqrt(cell_volumes[0]);

		surfDrainQKernal_calculation << <blocksPerGrid, threadsPerBlock >> > (inputKeyPtr.num_Junc,
			constVariables.g,
			constVariables.pi,
			dT,
			dx_surface,
			juncVariables.Qe_device,
			juncVariables.Qexchange_device,
			juncVariables.maxDepth_device,
			juncVariables.j_radius_device,
			juncVariables.junc_Area_device,
			juncVariables.h_j_device,
			juncVariables.zb_j_device,
			juncVariables.surfGrid_device,
			juncVariables.surfExGrid_device,
			inputKeyPtr.junc_expanValue,
			z.data.dev_ptr(),
			h.data.dev_ptr(),
			Q_surface.data.dev_ptr());
	}



	

	//----------------------------------------------------------------------------------------------------------------------
	__global__ void surfDrainQKernal_calculation( int num_Junc,
		 double g,
		 double pi,
		 double dT,
		 double dx,
		 double *Qe,
		double *Qdrainage,
		double *juncMaxDept,
		double *juncR,
		double *juncArea,
		double *hj,
		double *zbJunc,
		int *index,
		int *surfExGrid_device,
		int junc_expanValue,
		double *zbSurf,
		double *hs,
		double *QSurf) {

		// hs : water depth on the surface ground
		// hj: water depth of the junction in the drainage system
		// ind: the corresponding surface grid index
		// Q_s: exchange flow rate for the surface ground
		// Q_d: exchange flow rate for the drainage system


		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		double kesi=1e-10;
		double Ci = 1.0;
		
		double tempSingleQ;
		int juncSurfSize=junc_expanValue;



		while (tid< num_Junc)
		{
			Qe[tid]=0.00;

			for (int i=0; i<juncSurfSize; i++)
            {
            	int surf_ind=surfExGrid_device[tid*juncSurfSize+i];
            	//printf("surf_ind=%d\n", surf_ind);
            	QexchangeCalculation(hs[surf_ind], hj[tid], juncArea[tid], dx, Ci, pi, g, kesi, juncMaxDept[tid],
            		                 tempSingleQ, juncSurfSize, dT);
            	//printf("hs[%d]=%f, hj[%d]=%f, tempSingleQ=%f\n", surf_ind, hs[surf_ind]*10000000, tid, hj[tid]*10000000, tempSingleQ*10000000);
            	if (hj[tid] < juncMaxDept[tid])
                {
            		Q_maximumStorage_limit(Qe[tid], tempSingleQ, juncMaxDept[tid], hj[tid], juncArea[tid], 
            		                       dT, kesi, juncSurfSize);
                }

            	Qe[tid]=Qe[tid]+tempSingleQ;
            	QSurf[surf_ind]=-tempSingleQ/pow(dx,2);
            	
            }

            Qdrainage[tid] = Qe[tid]/juncArea[tid];
			
			tid += blockDim.x * gridDim.x;
		}

		
	}

//----------------------------------------------------------------------------------------------------------------------
    __device__ void  QexchangeCalculation(double hs, double hj, double juncArea, double dx, double Ci, double pi, double g,
    	                                  double kesi, double juncMaxDept, double &tempSingleQ, int juncSurfSize, double dT)
	{
		double junc_R=sqrt(juncArea/pi);
		if (dT<kesi)
		{
			tempSingleQ=0.0;

		}
		else
		{
			if (hs<kesi)	// no water at the surface grid
			{
				if ( hj<=juncMaxDept)
				{
					tempSingleQ=0.0;
				}
				else
				{
					
					tempSingleQ=-((hj-juncMaxDept)*juncArea/dT)/double (juncSurfSize);  // negetive flow: from drainage to surface
					
				}
			
			}
			else  // water exists in the surface grid
			{
				if (hj<juncMaxDept)  // positive
				{
					tempSingleQ=2.0 / 3.0*Ci*pi*junc_R*2.0*pow(2 * g, 0.5)*pow(hs, 3.0 / 2.0);
					if (tempSingleQ > hs*dx*dx/dT)
						tempSingleQ = hs*dx*dx/dT;
				}
				else if (hj==juncMaxDept)
				{
					tempSingleQ=0.0;
				}
				else  // water exists in the surface and water will flow out from junctins
				{
				
					tempSingleQ=-((hj-juncMaxDept)*juncArea/dT)/double (juncSurfSize);  // negetive flow
					
				}
			}
		}
		

	}


	//----------------------------------------------------------------------------------------------------------------------
    __device__ void QeLimitor(double &QeD, double dx, double hs, double juncArea, double hj, double dT, double maxDept){

		double Qmax1, Qmax2, Qmax3, Qmax4;

		// water from surface to drainage system condition 1:
		if (QeD>0 && hj < maxDept) 
		{
			if (QeD>hs*dx*dx/dT)
			{
				QeD=hs*dx*dx/dT;
			}
		}

		// water from surface to drainage system condition 2:
		if (QeD>0 && hj >= maxDept && hj < hs+maxDept)
		{
			Qmax1 = (hs+maxDept-hj)/(dT*(1/juncArea+1/pow(dx,2)));
			if (QeD> Qmax1)
			{
				QeD = Qmax1;
			}
		}

		// water from drainage system to water surface
		if(QeD<0 && hj>hs+maxDept)
		{
			Qmax2 = abs(hs+maxDept-hj)/(dT*(1/juncArea+1/pow(dx,2)));
			Qmax3 = hj*juncArea/dT;
			if (Qmax2 < Qmax3)
				Qmax4=Qmax2;
			else
				Qmax4=Qmax3;
			if (abs(QeD)>Qmax4)
				QeD=-Qmax4;
			
		}
		

		//printf("QeLimitor: hs= %f, dx=%f, juncArea=%f, hj=%f, Qe=%f\n",hs, dx, juncArea, hj, QeD);
	}

 //----------------------------------------------------------------------------------------------------------------------

	void surfH_limitator(cuFvMappedField<Scalar, on_cell>& h, int blocksPerGrid, int threadsPerBlock) {

		surfH_limilatorKernal << <blocksPerGrid, threadsPerBlock >> > (h.data.size(), h.data.dev_ptr());


	}

	//----------------------------------------------------------------------------------------------------------------------

	__global__ void surfH_limilatorKernal(unsigned int num_Girds, double *h_surf_device) {

		unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
		while (index  < num_Girds) {
			if (h_surf_device[index] < 0)
			{
				printf("Water depth in surfacel %d cell becomes negative ", index);
			}
			// if (index==154 || index==156 || index==196 || index==198)
        	//printf("surface waterdepth: cell %d, water: %f\n ", index, h_surf_device[index]);
			index += blockDim.x * gridDim.x;
		}

	}



	//----------------------------------------------------------------------------------------------------------------------
    __device__ void  Q_maximumStorage_limit(double &Qe, double &tempSingleQ, double juncMaxDept, 
    	                                    double hj, double juncArea, double dt, double kesi,
    	                                    int junc_expanValue){

    	if (dt<kesi)
		{
			tempSingleQ=0.0;

		}
		else
		{
			double Q_max_storage = (juncMaxDept - hj)*juncArea/dt;
	    	if (junc_expanValue == 1)
	    	{
	    		if (tempSingleQ > Q_max_storage)
	        		tempSingleQ = Q_max_storage;
	    	}
	    	else // junc_expanValue = 5, 9, 13
	    	{
	    		if (Qe + tempSingleQ > Q_max_storage)
	        	
	        		tempSingleQ = Q_max_storage - Qe;
	        	
	        	if (Qe == abs(Q_max_storage - kesi))
	        	
	        		tempSingleQ = 0.0;
	    	}    

		}
    	  	
        
    }


	//----------------------------------------------------------------------------------------------------------------------

	void junction_IntegrateQ(double dt, Input_information inputKeyPtr, Junc_variables &juncVariables,
		                      int blocksPerGrid, int threadsPerBlock){

		junction_IntegrateQKernel <<<blocksPerGrid, threadsPerBlock>>> (inputKeyPtr.num_Junc, 
			                                                           dt,
			                                                           juncVariables.h_j_device,
																       juncVariables.hn_j_device, 
																       juncVariables.Qexchange_device);

	}


	//----------------------------------------------------------------------------------------------------------------------

    __global__ void junction_IntegrateQKernel(int junc_number, double dt, double *h_j_device,double *hn_j_device, 
										 double* Qexchange_device){


	int tid= blockDim.x * blockIdx.x + threadIdx.x;

		while (tid<junc_number)   // in this function, N is the number of junction
		{

	       // add the water from surface
			
			h_j_device[tid]=h_j_device[tid]+(Qexchange_device[tid]*dt);
	        // printf("after adding surface water: dt=%f, Qexchange[%d]=%f, hn_j[%d]=%f \n", dt, tid, Qexchange_device[tid], 
	        // 	                                                                              tid, h_j_device[tid]);

			
			tid += blockDim.x * gridDim.x;
		}


	
	}
	//----------------------------------------------------------------------------------------------------------------------

	void Interface_gaugeOutput(double T,
    	cuFvMappedField<Scalar, on_cell>& h, 
		cuFvMappedField<Scalar, on_cell>& Q_surface,
		Input_information inputKeyPtr,
		Junc_attributes juncAttributes, 
		Junc_variables &juncVariables){

		 h.data.sync();
		 Q_surface.data.sync();
		 //juncVariables.synchronism(inputKeyPtr.num_Junc);
         //cudaMemcpy(juncVariables.Qexchange_device, juncVariables.Qexchange_out, inputKeyPtr.num_Junc*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(juncVariables.Qexchange_out, juncVariables.Qexchange_device, inputKeyPtr.num_Junc*sizeof(double), cudaMemcpyDeviceToHost);
	    
	     const char* directory = "output/";
	     std::string name = std::string(directory) + "InterfaceQ" + ".txt";
    
		 std::ofstream InterfaceQout;
         InterfaceQout.open(name.c_str(), std::ios::app);

         for (int i = 0; i<inputKeyPtr.num_Junc; i++)
         {

            InterfaceQout << T << "  ";
         	for (int j =0; j<inputKeyPtr.junc_expanValue; j++)
            {

            	int surf_index = juncAttributes.J_surfExGrid[i*inputKeyPtr.junc_expanValue+j];
            	//printf("surf_ind=%d\n", surf_index);
            	double Q_ = Q_surface.data.host_ptr()[surf_index];
            	double h_ = h.data.host_ptr()[surf_index];
            	InterfaceQout <<  surf_index <<"  " << h_ << "  " << Q_ << "  ";
            }
            InterfaceQout << "    "<< juncVariables.h_j_out[i] << "  " << juncVariables.Qexchange_out[i] << std::endl;
         }
         InterfaceQout << std::endl;
         InterfaceQout.close();

	}



}







