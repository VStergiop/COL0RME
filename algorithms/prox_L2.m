 function p = prox_L2(x, gamma, dir)
%function p = prox_L2(x, gamma, dir)
%
% This procedure computes the proximity operator of the function
%
%                    f(x) = gamma * ||x||_2
%
% When the input 'x' is an array, the computation can vary as follows:
%  - dir = 0 --> 'x' is processed as a single vector [DEFAULT]
%  - dir > 0 --> 'x' is processed block-wise along the specified direction
%
%  INPUTS
% ========
%  x     - ND array
%  gamma - positive, scalar or ND array compatible with the blocks of 'x'
%  dir   - integer, direction of block-wise processing


% default inputs
if nargin < 3 || (~isempty(dir) && dir == 0)
    dir = [];
end

% check input
sz = size(x); sz(dir) = 1;
if any( gamma(:) <= 0 ) || ~isscalar(gamma) && any(size(gamma) ~= sz)
    error('''gamma'' must be positive and either scalar or compatible with the blocks of ''x''')
end
%-----%


% linearize
sz = size(x);
if isempty(dir)
    x = x(:);
end
    
% compute the prox
xx = sqrt( sum(x.^2, dir) );
pp = max(0, 1 - gamma ./ xx);
p = bsxfun(@times, x, pp);

% revert back
p = reshape(p, sz);