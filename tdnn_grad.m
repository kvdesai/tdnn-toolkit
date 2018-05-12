function varargout = tdnn_grad(tdnn,Src, dEdO)
%  G = bpgrad(tdnn,Src,dEdO)
%  [G, gamma] = bpgrad(tdnn,Src,dEdO)
%  tdnn: structure with fields 'inputs','neurons','links','uO'
%  Src: Source matrix containing all threshold layer, inputs, and neuron 
%     outputs for all time-steps [Nu x Nt]. Arranged as [b; inputs; Y].
%     Typically the 2nd output argument of 'tdnn_fwpass'.
%  dEdO:Derivative of the objective func wrt each output [No x Nt]
%  G: A column vector containing gradient of each link weigth [Nl x 1]
%  gamma: Error basket for each neuron at each time-step [NN x Nt]
%  NOTE: Contribution of prehistory towards gradient vector is ignored.

%% Algorithm summary
% 1. Fill output unit errors by the derivative of the objectiveFcn.
% 2. Start from the output units and repeat following until 
%    we get to the bottom last unit
%    a. Scale the errors accumulated by the derivative of the transferFcn,
%       evaluated at time n, i.e, Yj(n)*(1-Yj(n))
%    b. Find units that feed to the current unit
%    c. Propagate the error of the current unit to each
%       feeder by applying the weight and delay of the link to the feeder
%       Go to the next unit 
% 3. Loop through all links and fill the gradient vector by
%   dedW_ijd

% mydiary('Computing Gradient');
uO = reshape(tdnn.uO,1,[]); %Index of output units
NI = tdnn.inputs.num; %Number inputs
NN = length(tdnn.neurons); %Number of Neurons
NO = length(uO); %Number of output neurons

LF = tdnn.links(:,1); %List of Link source units ("From" Units)
LT = tdnn.links(:,2); %List of Link destiation units ("To" Units)
LD = tdnn.links(:,3); %List of Link delays
LW = tdnn.links(:,4); %List of Link weights

%% Sanity check
if ~isequal(size(dEdO,1),NO)
    error('Violated: No. of rows of derivative matrix = No. of outputs');
end
if ~isequal(size(Src,1),NN+NI+1)
    error('Violated: No. of rows of src matrix = 1 + inputs + No. of neurons');
end
if ~isequal(size(Src,2),size(dEdO,2)+max(LD))
    error('Violated: No. of columns of src matrix = No. of columns of deriv matrix + Max Delay');
end

Nt = size(Src,2); %Number of time steps
iTpos = [max(LD)+1:1:Nt];%Indices for which the time is >= 0

%% Core code
gamma = zeros(NN,Nt); %Error basket for each neuron at each time step
% STEP 1: Fill in the erro baskets for output neurons
for c = 1:NO
    gamma(uO(c)-NI-1,iTpos) = dEdO(c,:);
end
clear dEdO;

% STEP 2: Fill in all the error baskets
for k = NN:-1:1 %processing each neuron from top down
    % STEP 2.a: Scale the accumulated error by the derivative of the Xferfun
    dYk = transferfun(Src(1+NI+k,:),tdnn.neurons(k).xferfun,'deriv');
    gamma(k,:) = gamma(k,:) .* dYk;
    
    % STEP 2.b: Find the links from the feeder neurons (not feeder inputs)
    Lbase = 1:length(LT);
    iL = Lbase(and(logical(LT == 1+NI+k), logical(LF > NI + 1)));
         %Link index of the links from the feeder neurons
    iN = LF(iL) - (NI + 1);
         %Neuron index of the links from the feeder neurons    
    % STEP 2.c: Back-propagate the error onto the feeder neurons
    for n = 1:length(iL)
        gamma(iN(n),iTpos-LD(iL(n))) = gamma(iN(n),iTpos-LD(iL(n))) + ...
                                    gamma(k,iTpos) * LW(iL(n));
    end      
end

% STEP 3
G = zeros(length(LW),1);

for c = 1:length(LW) %Processing each link one by one
    %Grad_(s,u,d) = sum_t[gamma_(u,t) .* src_(s,t-d)]
    G(c,1) = sum( gamma(LT(c)-NI-1,iTpos) .* Src(LF(c),iTpos - LD(c)) );
end

varargout{1} = G; %Gradient vector containing gradient for each link
if nargout > 1
    varargout{2} = gamma; %If needed for debugging
end
clear gamma Src LF LT LD LW %surprizingly, this helps! Memory leak?

