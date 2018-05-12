#include <math.h>
#include "mex.h"

/// Compiler directives to make Mex array indexing a bit clear
///  CAUTION: Both operands of a product must be in parenthesis
#define ELE2(Arr,i,j,M,N) (*(Arr + (j)*(M) + i))
#define ELE3(Arr,i,j,k,M,N,P) (*(Arr + ((k)*(N) + j)*(M) + i))

double transferfun(double x, int type)
{	
	double y;
	switch( type ) 
	{
		case 2: //tanh			
			y = (2.0 / (1.0 + exp(-2.0*x))) - 1.0;				
			break;
			
		case 1: //logsig			
			y = 1.0 / (1.0 + exp(-x));				
        	break;
						
		case 0: //linear			
			y = x;        	
			break;
				
		default: //error
			mexErrMsgTxt("xferfun: Unrecognized type");				
	}
	return y;
}

double transferfun_deriv(double x, int type)
{	
	double y;
	switch( type ) 
	{
		case 2: //tanh		
			y = 2.0*(1.0 + x) * (1.0 - x);
			break;
				
		case 1: //logsig					
			y = x * (1.0 - x);
		    break;
			    
		case 0: //linear			
			y = 1.0;
			break;
				
		default: //error
			mexErrMsgTxt("xferfun_deriv: Unrecognized type");
			
	}
	return y;	
}

//----------- Fwd Pass through TDNN --------NO ERROR CHECKING
int tdnn_fwpass(double *Tlink, mwSize *Tnu, mwSize *TDims, double *InMat, mwSize *InDims, double *OutMat)
// First four arguments describe the TDNN (Inputs, Links, Neurons, Dims)
// Tin [1 x 4]: [NI, MultScale, Ndelay, Nnan];
// Tlink[NL x 5]: [Lfrom, Lto, Ldelay, Lweight, Lfrozen]
// Tnu[NN x 4]: [Output?, Xferfun, Ndelay, Nnan]
// 			Xferfun: 0=linear, 1=logsig, 2 = tansig
// TDims [1 x 5]:  [NI, NL, NN, NO, MaxD] as mwSize
// InMat [NI x NT x NX]: Inputs supplied in 1D doubles array
// InDims [1 x 3] : [NI, NT, NX] as mwSize array
// NI = Number of input channels, Must match with Tin[0]
// NT = Number of timesteps per example
// NX = Number of examples
// 
// OutMat [(1+NI+NN) x (MaxD+NT) x NX] - contains values of all units at all timesteps (including prehistory)
// 		MaxD is the maximum delay among all links. Bias unit, inputs, and neurons all are "units"
// Note: it is assumed that the pointer to the OutMat has been initialized 
//		to point to appropriately allocated array
{
	int res = 0;
	register int ii, it, ix, il;
	register int NI, NT, NX, NL, MaxD, D0Max, D1Max; 
	register int iSrc, iDest, nDel;
	
	
	double Wgt; 		
	int NN, NO;
	int *bAppTF; ///Pointer to store boolean for each link whether to apply TF 
			
	NI = InDims[0];
	NT = InDims[1];
	NX = InDims[2];
	
	///Checking argument consistency
	if(NI != TDims[0])
		mexErrMsgTxt("tdnn_fwpass: Inconsistent arguements for NI");
		
	NL = TDims[1];
	NN = TDims[2];
	NO = TDims[3];
	MaxD = TDims[4];
		
	///------------------ Initialize OutMat --------------------
	/// OutMat [(1+NI+NN) x (MaxD+NT) x NX]
	/// 	OutMat(0,:,:) = 1 -->the bias unit in the first row
	/// 	OutMat(1 to NI, MaxD to End , :) = InMat 
	/// 	Rest is 0.
	
	ii = 0; ///Setting the bias unit
	D0Max = 1+NI+NN;
	D1Max = MaxD + NT;
	for(it = 0; it < D1Max; it++)
	{
		for(ix =0; ix < NX; ix++)
		{
			ELE3(OutMat, ii,it,ix, D0Max, D1Max, NX) = 1.0;
		}
	}
	/// Bias unit set, start from the 2nd row
	for(ii = 1; ii < D0Max; ii++)
	{
		for(it = 0; it < D1Max; it++)
		{
			for(ix =0; ix < NX; ix++)
			{
				if((ii < NI+1) && (it >= MaxD)) ///InMat
					ELE3(OutMat, ii,it,ix, D0Max, D1Max, NX) = ELE3(InMat,ii-1,it-MaxD,ix, NI, NT, NX);
				else
					ELE3(OutMat, ii,it,ix, D0Max, D1Max, NX) = 0.0;
					
			}
		}
	}	
	
	/// --------Make a binary flag for links at which to apply transferfun			
	bAppTF = mxCalloc(NL,sizeof(int));	
	for(il = 0; il < NL-1; il++)
	{
		if(ELE2(Tlink,il,1,NL,5) < ELE2(Tlink,il+1,1,NL,5))
			bAppTF[il] = 1;
		else
			bAppTF[il] = 0;
	}
	bAppTF[NL-1] = 1; ///Always apply TF after processing the last link
	
	{///---------------- Core Processing ---------------
		
		
		for(ix = 0; ix < NX; ix++) ///Loop through examples
		{		 
			for(il = 0; il < NL; il++) ///Loop through links
			{	
				/// Grab source and destination index, delay, and weight
				iSrc = (int)ELE2(Tlink,il,0,NL,5) -1;
				iDest = (int)ELE2(Tlink,il,1,NL,5) -1;
				nDel = (int)ELE2(Tlink,il,2,NL,5);				
				Wgt = ELE2(Tlink,il,3,NL,5);
				//mexPrintf("ix: %d, il:%d, iSrc: %d, iDest: %d, nDel: %d, Wgt:%f\n",ix,il,iSrc,iDest, nDel, Wgt);
				
				/// For all positive time points, add activation from the link source 
				/// to the total activation for this neuron. Apply TF to the result if we should	
								
				/// The matlab code below would require two for loops each with length = length of iTpos				
				// 		src(iDest,iTpos) = src(iDest,iTpos) + Wgt*src(iSrc,iTpos-nDel); %$$
				// 		if any(ibreak == c)
				// 			src(iDest,iTpos) = transferfun(src(iDest,iTpos),net.neurons(iDest-NI-1).xferfun,'tf'); %$$
				// 		end
				/// But because we know that a unit can appear in the Dest column *only after* all of its
				/// source units have been computed, we can combine the two statements in a single for loop.
				
				if(bAppTF[il] == 1) ///Apply TF 
				{
					int tftype = ELE2(Tnu,iDest-NI-1,1,NN,4);
					//mexPrintf("TFtype: %d\n",tftype);
					for(it=MaxD; it < MaxD+NT; it++) ///iTpos
					{
						/// Add to the Activation value
						ELE3(OutMat, iDest,it,ix, D0Max, D1Max, NX)  += 
								(Wgt) * ELE3(OutMat, iSrc,it-nDel,ix, D0Max, D1Max, NX);
						/// Apply TF
						ELE3(OutMat, iDest,it,ix, D0Max, D1Max, NX) = 
							transferfun(ELE3(OutMat, iDest,it,ix, D0Max, D1Max, NX), tftype);
					}	
				}				
				else
				{
					for(it=MaxD; it < MaxD+NT; it++) ///iTpos
					{
						/// Add to the Activation value
						ELE3(OutMat, iDest,it,ix, D0Max, D1Max, NX)  += 
								(Wgt) * ELE3(OutMat, iSrc,it-nDel,ix, D0Max, D1Max, NX);
					
					}					
				}				
			}///Finished processing all links				
		}/// Finished processing all Examples
	} ///End of core processing Block
	
	/// Deallocate the arrays we allocated. Actually Mex compiler is smart enough to do this automatically.
	mxFree(bAppTF);
	return res;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	
	double *Tlink, *Inmat, *Outmat;
	mwSize *Tnu, *Tdims, *Indims, dims[3];
	
	
	///Inputs: Tlink, Tnu, Tdims, InMat, InDims, MxArray Pointer to OutMat
	Tlink = mxGetPr(prhs[0]);
	Tnu = mxGetPr(prhs[1]);
	Tdims = mxGetPr(prhs[2]);
	Inmat = mxGetPr(prhs[3]);
	Indims = mxGetPr(prhs[4]);
		
	/// Compute dimensions of the output matrix
	dims[0] = 1 + Tdims[0] + Tdims[2];///D0Max = 1+NI+NN;
	dims[1] = Tdims[4] + Indims[1]; ///D1Max = MaxD + NT;
	dims[2] = Indims[2]; ///NX
	
	//mexPrintf("Output dims: [%d x %d x %d]\n", dims[0], dims[1], dims[2]);
	/* create a given dimensional array of doubles */
	plhs[0] = mxCreateNumericArray(3,dims,mxDOUBLE_CLASS,mxREAL);
	Outmat = mxGetPr(plhs[0]);
	
	//mexPrintf("Allocated array, now calling tdnn_fwpass()...\n");
	tdnn_fwpass(Tlink, Tnu, Tdims, Inmat, Indims, Outmat);
	//mexPrintf("Returned from tdnn_fwpass()...\n");
}


// void transferfun_vec(double* x, int xlen, double* y, int type)
// {	
	// register unsigned int c;
	// switch( type ) 
	// {
		// case 2: //tanh		
			// for(c = 0; c < xlen; c++)
				// y[c] = (2.0 / (1.0 + exp(-2.0*x[c]))) - 1.0;				
			// break;
			
		// case 1: //logsig		
			// for(c = 0; c < xlen; c++)
				// y[c] = 1.0 / (1.0 + exp(-x[c]));				
        	// break;
			
		// case 0: //linear
			// for(c = 0; c < xlen; c++)
				// y[c] = x[c];				
        	// break;			
				
		// default: //error
			// mexErrMsgTxt("xferfun: Unrecognized type");				
	// }	
// }

// void transferfun_vec_deriv(double* x, int xlen, double *y, int type)
// {	
	// register unsigned int c;
	// switch( type ) 
	// {
		// case 2: //tanh
		
			// for(c = 0; c < xlen; c++)
				// y[c] = 2.0*(1.0 + x[c]) * (1.0 - x[c]);
				
			// break;
		// case 1: //logsig
		
			// for(c = 0; c < xlen; c++)
				// y[c] = x[c] * (1.0 - x[c]);
				
        	// break;
		// case 0: //linear
			// for(c = 0; c < xlen; c++)
				// y[c] = 1.0;
			// break;
				
		// default: //error
			// mexErrMsgTxt("xferfun_deriv: Unrecognized type");
			
	// }	
// }
// void mexFunction_tranferfun_vec_test(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
// {
  // double *x,*y, *yd;
  // int type;
  // mwSize mrows,ncols;
  
  // /*  check for proper number of arguments */
  
  // if(nrhs!=2) 
    // mexErrMsgTxt("Two inputs required.");
  // if(nlhs!=2) 
    // mexErrMsgTxt("Two output required.");
    
  // /* check to make sure the first input argument is double vector */
  // if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetM(prhs[0]) !=1 ) {
    // mexErrMsgTxt("Input x must be a double, row vector.");
  // }
  // /* check to make sure the second input argument is integer scalar */
  // if( !mxIsInt32(prhs[1]) || mxIsComplex(prhs[1]) ||
      // mxGetN(prhs[1])*mxGetM(prhs[1])!=1 ) {
    // mexErrMsgTxt("Input 'type' must be an integer scalar.");
  // }
  
   // /*  create a pointer to the input matrix x */
  // x = mxGetPr(prhs[0]);
  
  // /*  get the dimensions of the input matrix x */
  // mrows = mxGetM(prhs[0]);
  // ncols = mxGetN(prhs[0]);
  
  // /*  get the scalar input type */
  // type = mxGetScalar(prhs[1]);
  
  // /*  set the output pointer to the output matrix */
  // plhs[0] = mxCreateDoubleMatrix(mrows,ncols, mxREAL);
  // plhs[1] = mxCreateDoubleMatrix(mrows,ncols, mxREAL);
  
  // /*  create a C pointer to a copy of the output matrix */
  // y = mxGetPr(plhs[0]);
  // yd = mxGetPr(plhs[1]);
  
  // /*  call the C subroutine */
  // transferfun_vec(x,ncols,y,type);
  // transferfun_vec_deriv(x,ncols,yd,type);  
// }
