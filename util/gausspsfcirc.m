function [lambda, G, Sm] = gausspsfcirc(I, pixsz, srf, fwhm, isnormalized)

% Should we normalize columns to have unitary peak?
if nargin < 5
    isnormalized = false;
end

% Full-width at half-maximum of a Gaussian;
sig = fwhm / (2*sqrt(2*log(2)));

% Original dimension
L = I * srf;

if mod(L,2) == 0
    ii = (-L/2+1):(L/2);
else
    ii = (-(L-1)/2):((L-1)/2);
end
g = normcdf((pixsz/srf) * ii, pixsz/(2*srf), sig) - ...
    normcdf((pixsz/srf) * (ii-1), pixsz/(2*srf), sig);    
g = fftshift(g);

% Optional normalization step
if isnormalized
    g = g/max(abs(g));
end

% Filter frequency response
lambda = fft(g);

% Corresponding matrix
if nargout > 1
    G = toeplitz([g(1) fliplr(g(2:end))], g);
    G = reshape(G',[L srf I]);
    G = squeeze(sum(G, 2))';
end

if nargout > 2
    % Selection matrix
    Sm = repelem(eye(I), 1, srf);    
end