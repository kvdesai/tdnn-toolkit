function y = transferfun(x,method,mode)
if nargin < 3 || isempty(mode), mode = 'tf'; end
mode = lower(mode);
if isequal(mode,'tf')
    switch lower(method)
        case 'logsig'
            y = 1 ./ (1 + exp(-x));
        case 'tansig'
            y = (2 ./ (1 + exp(-2*x))) - 1;
        case 'linear'
            y = x;
        otherwise
            error('Unrecognized option for tranfer function');
    end
elseif isequal(mode,'deriv')
    switch lower(method)
        case 'logsig'
            y = x .* (1 - x);
        case 'tansig'
            y = 2*(1 + x) .* (1 - x);
        case 'linear'
            y = ones(size(x)); 
        otherwise
            error('Unrecognized option for tranfer function');
    end
else
    error(['Unrecognized mode: ',mode,'. Usage: [{''tf''},''deriv'']']);
end