 function p = project_box(x, low, high)
%function p = project_box(x, low, high)
%
% The procedure compute the projection onto the constraint set:
%
%                     low <= x <= high
%
% When the input 'x' is an array, the output 'p' is computed element-wise.
%
%  INPUTS
% ========
%  x    - ND array
%  low  - scalar or ND array with the same size as 'x'
%  high - scalar or ND array with the same size as 'x'


% check inputs
if ~isscalar(low) && any(size(low) ~= size(x))
    error('''low'' must be either scalar or the same size as ''x''')
end
if ~isscalar(high) && any(size(high) ~= size(x))
    error('''high'' must be either scalar or the same size as ''x''')
end
if any( low(:) >= high(:) )
    error('''low'' must be lower than ''high''')
end
%-----%


% compute the projection
p = min(x, high);
p = max(p, low);