function n = norm_nuclear(x)
%NORM_NUCLEAR - Nuclear norm of x
%   Usage: norm_nuclear(x) 
%  
%   Input parameters
%       x       : a matrix
%   Output parameters
%       n       : nuclear norm of x


if issparse(x)
    n = sum(svds(x, min(size(x))));
else
    n = sum(svd(x, 'econ'));
end

end

