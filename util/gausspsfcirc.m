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

% % The airy disc model of the psf
% ll=linspace(-0.5,0,L/2+1);
% lr=linspace(0,0.5,L/2);
% [X,Y]=meshgrid([ll,lr(2:end)],[ll,lr(2:end)]);
% [th,rho]=cart2pol(X,Y);
% OTF=fftshift(1/pi*(2*acos(abs(rho)/fc)-sin(2*acos(abs(rho)/fc))).*(rho<fc));
% psf=real(ifft2(OTF));
% g =psf(1,:);
% g = g/sum(g(:));

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