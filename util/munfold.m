function T = munfold(T,irow,icol,dims)
% MUNFOLD(T,irow,icol,dims) rearrages the entries of tensor T into a matrix 
% according to the partition of its N modes into two sequences (Matlab 
% vectors), irow and icol. These sequences specify which modes are 
% associated with the rows and columns of the resulting matrix, and in 
% which order.

% Optionally, the dimensions of T can be passed as a fourth argument.
%
% The arguments irow and icol must satisfy:
%       length([irow icol]) == N 
%       sort([irow icol]) == 1:N 
%
% tenslib v1  - jan 2019
% J. H. de M. Goulart
%
% See also MFOLD

if nargin < 4
    dims = size(T);
end

irow = fliplr(irow(:)');
icol = fliplr(icol(:)');

N = length(dims);

if length(irow)+length(icol) ~= N
    error('Incorrect specification of unfolding modes!');
end

if any((sort([irow icol]))-(1:N) ~= 0) 
    error('Incorrect specification of unfolding modes!');
end    
    
T = permute(T,[irow icol]);
T = reshape(T,[prod(dims(irow)) prod(dims(icol))]);
