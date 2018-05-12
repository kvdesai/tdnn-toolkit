function newnet = tdnn_computeshift(tdnn)
% Computes the sample-shift (Nshift) at the output of each neuron of the given network.
% This is the number of samples by which the output of the network is
% delayed in time (to right), compared to the original input vector.
% The targets of the network must be shifted accordingly. Also, because the
% sample-shift also means the length of the processing window, half of
% Nshift samples at each end of the target vector must be set to NaN.
% Updates Nnan (=NnanL=NnanR) , and Ndelay fields of each neuron.

Nnan_pre = tdnn.inputs.Nnan;
Ndelay_pre = tdnn.inputs.Ndelay;
NI = tdnn.inputs.num; %Number of inputs
for iN = 1:length(tdnn.neurons)
    clear feeder sNshift;
    % Find links feeding to this neuron
    links = tdnn.links(logical(tdnn.links(:,2) == iN + NI + 1),:);
    % Find feeder neurons if any
    lfeeders = logical(links(:,1) > NI + 1); 
    iSn = links(lfeeders,1) - (NI + 1);
    
    % Compute for each link: Delay for the link + Delay at the source
    dN_total = zeros(size(links,1),1);
    Nnan_total = zeros(size(links,1),1);
    for lc = 1:size(links,1)
        % First Determine the delay at the feeder units
        if lfeeders(lc) % the feeder is a neuron
            iN_feeder = links(lc,1) - NI - 1;
            dN_feeder = tdnn.neurons(iN_feeder).Ndelay;
            Nnan_feeder = tdnn.neurons(iN_feeder).Nnan;
        else % the feeder is a network input unit
            dN_feeder = Ndelay_pre;
            Nnan_feeder = Nnan_pre;
        end
        dN_me = links(lc,3); %Delay at the link
        Nnan_me = ceil(dN_me*0.5);
        
        dN_total(lc,1) = dN_me + dN_feeder;
        Nnan_total(lc,1) = Nnan_me + Nnan_feeder;       
    end
    
    % Find the feeder link with the maximum total delay
    isuper = find(dN_total == max(dN_total),1,'first');
    
    tdnn.neurons(iN).Ndelay = dN_total(isuper,1);
    tdnn.neurons(iN).Nnan = Nnan_total(isuper,1);
end

newnet = tdnn;
