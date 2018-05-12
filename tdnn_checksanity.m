function out = tdnn_checksanity(net)
% verifies that the given network complies with the network syntax.
% Currently expects syntax version 3.5 and above only.
netfields = {'inputs','neurons','links','uO','exptype','syntaxver','MulFactors'};
neuronfields = {'Nnan','Ndelay','Out','xferfun'};
inputsfields = {'num','Nnan','Ndelay',};
NC = 5; %Number of columns in the links matrix

gotnetfields = fieldnames(net);
for c = 1:length(netfields)
    if ~any(strcmp(gotnetfields,netfields{c}))
        out = 1;
        error([netfields{c},' field is missing in the provided network']);
    end
end

gotinpfields = fieldnames(net.inputs);
for c = 1:length(inputsfields)
    if ~any(strcmp(gotinpfields,inputsfields{c}))
        out = 2;
        error([inputsfields{c},' subfield is missing in .inputs']);
    end
end

for cn = 1:length(net.neurons)
    gotnufields = fieldnames(net.neurons(cn));
    for c = 1:length(neuronfields)
        if ~any(strcmp(gotnufields,neuronfields{c}))
            out = 3;
            error(sprintf('%s subfield is missing in .neurons(%d)',neuronfields{c},cn));
        end
    end
end

if ~isequal(size(net.links,2),NC)
    out = 4;
    error(['.links matrix must have ',num2str(NC),' columns']);
end

ibad = find(net.links(:,2) <= net.inputs.num + 1);
if ~isempty(ibad)
    out = 5;
    error(['None of the Input units or the Thresholding unit can ever be a ''To Unit''.',...
           'Bad links: ',num2str(ibad)]);    
end

ibad = find(net.links(:,2) - net.links(:,1) <= 0);
if ~isempty(ibad)
    out = 6;
    error(['A neuron indexed later cannot feed itself or a neuron indexed earlier', ...
           'Bad links: ',num2str(ibad)]);
end

ibad = find(net.links(2:end,2) - net.links(1:end-1,2) < 0);
if ~isempty(ibad)
    out = 7;
    error(['Links must be ordered such that the ''To unit'' index ',...
        'never decreases. Bad links: ',num2str(ibad)]);
end

% Now check if the output unit indexes are valid
iO = net.uO - net.inputs.num - 1;
if max(iO) > length(net.neurons) || min(iO) < 1
    out = 8;
    error('The list of output neurons is bad');
end

% Now check if MulFactors is exactly 2 elements
if numel(net.MulFactors) ~= 2
    error('MulFactors field must have exactly two elements: Input Factor, Output Factor');
end
