function varargout = objectivefun(O,T,W)
% Computes objective function and (if asked) its derivative wrt outputs
% USAGE:
%       
%               E = objectivefun(O,T,W)
%       [E, dEdO] = objectivefun(O,T,W)
% [E, dEdO, eraw] = objectivefun(O,T,W)
% Where,
% O = Outputs of the network (without prehistory) [No x Nt]
% T = Targets  [No x Nt]
% W = Weights to be applied to each error [No x Nt]

% E = Mean-squared Error (or the Objective Function) [1 x 1]
% dEdO = Analytically evaluated derivatives of error wrt each output [No x Nt]
% eraw = Raw error (unweighted diff) for each output at each time step [No x Nt]

% mydiary('Computing Objective Function');
if ~isequal(size(O),size(T),size(W))
    error('The size of Output matrix, Target matrix, and weight matrix must be identical');
end

eraw = zeros(size(T));
igood = logical(W ~= 0);
eraw(igood) = T(igood) - O(igood);

Nvalid = nnz(W);
E = (1/Nvalid)*sum(sum(W.*(eraw.^2)));

varargout{1} = E;
if nargout > 1
    varargout{2} = eraw.*W*(-2/Nvalid);
end
if nargout > 2
    varargout{3} = eraw;
end
clear eraw O T