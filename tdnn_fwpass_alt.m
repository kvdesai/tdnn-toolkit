function varargout = tdnn_fwpass(net,P)
% Run a forward pass through the given network with given inputs
% Usage:        Y  = tdnn_fwpass(net,P);
% where,
% net = dynamic neural network 
% P = input matrix [I x T], I = number of elements in input vector
% Y = output matrix [O x T], O = number of output neurons in the network
% Kalpit Desai

if ~isequal(size(P,1),net.inputs.num)
    error(['The given network expects ',num2str(net.inputs.num),...
        ' rows in the input matrix']);
end

% if isfield(net.inputs,'MultScale')
%     P = P * net.inputs.MultScale;
% end
Nt = size(P,2); %Number of time steps
Nl = size(net.links,1); %Number of links
uO = reshape(net.uO,1,[]); %Index of output units
NI = net.inputs.num; %Number inputs
NN = length(net.neurons); %Number of Neurons
NO = length(uO); %Number of output neurons

LF = net.links(:,1); %List of Link source units ("From" Units)
LT = net.links(:,2); %List of Link destiation units ("To" Units)
LD = net.links(:,3); %List of Link delays
LW = net.links(:,4); %List of Link weights

MaxD = max(LD); % Maximum delay
preh = zeros(NI,MaxD); % Prehistory
inp = [preh,P]; %Prepend inputs with the prehistory
inp = [ones(1,size(inp,2));inp]; %Put the bias source at the first raw

% % Fill zeros for outputs of each Neuron
% Y = zeros(length(net.neurons),size(inp,2));
% % Put the neuron outputs at the end of the source matrix
% src = [inp;Y(1:end,:)]; 
src = [inp;zeros(NN,size(inp,2))]; %$$
clear inp preh P; %THIS HELPS!
%Somehow matlab seems to hide these variables upon returning, but doesn't 
%free-up the associated memory. There seems to be a memory leak!!

iTpos = [1:Nt]+MaxD;%Indices of src for which the time is >= 0
% Determine links for which transfer function is to be applied
% Assumes that link-list is ordered such that a neuron appears in the Dest column
% only after all its feeding neurons have appeared in the Dest column. This (and
% other assumptions) are actually verified by tdnn_checksanity 
ibreak = find(LT(1:end-1) < LT(2:end));
ibreak = [reshape(ibreak,1,[]),Nl];%Allways apply TF at the last unit
for c = 1:Nl %Process each link one by one
    iU = LT(c)-NI-1; %Index of the current neuron    
    % Y(iU,iTpos) = Y(iU,iTpos) + LW(c)*src(LF(c),iTpos-LD(c));
    src(LT(c),iTpos) = src(LT(c),iTpos) + LW(c)*src(LF(c),iTpos-LD(c)); %$$
    if any(ibreak == c)    
        % Apply transfer function to the total input / activation of this neuron
        % Y(iU,iTpos) = transferfun(Y(iU,iTpos),net.neurons(iU).xferfun,'tf');
        % % Update the source matrix to accomodate the output of this neuron
        % src(LT(c),:) = Y(iU,:);        
        src(LT(c),iTpos) = transferfun(src(LT(c),iTpos),net.neurons(iU).xferfun,'tf'); %$$
    end
end
% varargout{1} = Y(uO,iTpos); %Output neurons
varargout{1} = src(uO,iTpos); %$$
if nargout > 1
    varargout{2} = src; %Complete Source Matrix including prehistory
end
clear Y src;