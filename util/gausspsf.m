function [g2D] = gausspsf(I, pixsz, srf, fwhm)

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

g2D = reshape(kron(g,g),[L L]);

% % From the definition
% h = @(x, y, mu, sig, I) I * exp(-((x-mu).^2 + (y-mu).^2)/2/sig^2); 
% jj = (-L/2):(L/2);
% [x, y] = meshgrid(jj*(pixsz/srf),jj*(pixsz/srf));
% psf = h(x, y, 0, sig, 1);
% psf = 1/(sum(sum(psf(1:end-1,1:end-1))))*psf(1:end-1,1:end-1);

