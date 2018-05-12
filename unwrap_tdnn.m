function T = unwrap_tdnn(tdnn)
% Takes a given tdnn structure and unwraps it in four arrays

inputs = [tdnn.inputs.Ndelay;
          tdnn.inputs.Nnan];% [Num, MultScale, Ndelay, Nnan]

links = tdnn.links; %links are already in matrix format [Nlinks x 5]


neurons = -1*ones(length(tdnn.neurons),4); % [Nn x 4] - [Output?, Xferfun, Ndelay, Nnan]
for c = 1:size(neurons,1)
    neurons(c,1) = tdnn.neurons(c).Out;
    % $$$ This mapping must be preserved in the mex file
    switch tdnn.neurons(c).xferfun
        case 'linear'
            neurons(c,2) = 0;                
        case 'logsig'
            neurons(c,2) = 1;
        case 'tansig'
            neurons(c,2) = 2;
        otherwise
            error('Unrecognized xferfun %s',tdnn.neurons(c).xferfun);    
    end
    neurons(c,3) = tdnn.neurons(c).Ndelay;
    neurons(c,4) = tdnn.neurons(c).Nnan;    
end
if any(neurons(:) < 0)
    error('what?');
end

T.inputs = inputs;
T.neurons = int32(neurons);
T.links = links;
T.dims = int32([tdnn.inputs.num,...
                size(links,1), ...
                size(neurons,1), ...
                length(tdnn.uO), ...
                max(links(:,3))]);
T.uO = tdnn.uO;
T.MulFactors = tdnn.MulFactors;
T.info = {tdnn.exptype, ['UnWrapped',tdnn.syntaxver]};

end