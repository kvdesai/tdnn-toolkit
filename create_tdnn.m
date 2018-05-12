function tdnn = create_tdnn(neurons,sets)
% Creates link list based on neuron definitions and other parameters.
% Packages the link list into a complete network structure and outputs. 

syntaxver = '3.5';%Version of the network architecture
tdnn = struct('inputs',[],'neurons',[],'links',[],'uO',[],...
    'exptype',[],'syntaxver',[],'MulFactors',[]);

NI = sets.inputs.num;
W = sets.W;
b = sets.b;
D = sets.D;
uS = sets.uS;
fz = sets.fz;
tdnn.syntaxver = syntaxver;
tdnn.exptype = sets.exptype;
tdnn.inputs = sets.inputs;
tdnn.MulFactors = sets.MulFactors;

NUNITS = length(neurons);
% Units: [Bias, Inputs, Neurons] => [1, 2:numinputs+1, numinputs+2: NS]
uB = 1; %Unit index of Threshold unit
uIn = reshape(2:NI+1,[],1); %Unit index of inputs
uN = reshape((NI+1)+1:(NI+1)+NUNITS,[],1);% Unit index of Neurons

iO = [];
for c = 1:NUNITS
    if neurons(c).Out == 1
        iO = [iO,c];
    end
    N(c) = length(W{c}); %Number of links other than bias
end
tdnn.uO = uN(iO); %Unit index of output units 
tdnn.neurons = neurons;
NL = sum(N) + length(neurons); %Total Links = weights + biases

links = zeros(NL, 5); links(:) = NaN;
ilast = 0;
for c = 1:length(tdnn.neurons)    
    %Link Syntax:    [SourceUnit,         DestUnit,    Delay,  Weight, Frozen?]
    links(ilast+1,:) =  [uB,                  uN(c),       0,  b(c),   0]; %Bias
    links(ilast+2:ilast+N(c)+1,:) =...
                        [uS{c},   ones(N(c),1)*uN(c),   D{c},  W{c}, fz{c}];
    ilast = ilast+N(c)+1;
end
tdnn.links = links;
% Now compute and update Nshift and Nnan fields for each neuron
tdnn = tdnn_computeshift(tdnn);
tdnn_checksanity(tdnn);