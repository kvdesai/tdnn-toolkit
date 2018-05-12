function [op,Targs,eW] = decimate_netinputs(nnet,din,Targets,eWght)
% Decimate the preprocessed data to match the input resolution of the
% chosen network
% Wanted to use 'decimate' instead of 'resample'. 'resample' needs integer values
% of P and Q to produce the new sampling rate of P/Q times original.
% But decimate only works on a vector, not matrix. So
% first making sure that NNI_resolution (ms) is an integral multiple of
% the resolution of preprocessed data, and then providing the ratio as
% Q with P = 1;

res_proc = din.res_ms;
res_nni = nnet.inputs.res_ms;
if mod(res_nni, res_proc)
    error('NNI resolution must be an integral multiple of preprocessing resolution');
else
    Rdown = res_nni/res_proc;
end
op = (resample(din.op',1,Rdown))';
% For the targets and error weight function, simply keep every Rdown-th
% point. Because these are discretely varying functions to begin with.
eW = eWght(:,1:Rdown:end);
Targs = Targets(:,1:Rdown:end);    