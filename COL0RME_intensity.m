%% Build dictionary -- 2D and remove boundaries

% Coarse grid size
N = size(Y_nobound,1); 

% Super-resolution grid size
L = N * srf;

% Build dictionary
PSF = gausspsf(N, pixsiz, srf, fwhm);

PSFfft = fft2(fftshift(PSF));
PSFconj = conj(PSFfft);

Mech = zeros(N, L);
for i = 0 : N-1
    Mech(i+1, 1+srf*i : srf+srf*i) = 1;
end
MechT = Mech';

is = ones(L);
is(supp_x>0) = 0;

D.Dh_DFT = psf2otf([1,-1],[L L]);
D.Dv_DFT = psf2otf([1;-1],[L L]);
D.DhT_DFT = conj(D.Dh_DFT);
D.DvT_DFT = conj(D.Dv_DFT);

D.Dh_DFT_b = psf2otf([1,-1],[N N]);
D.Dv_DFT_b = psf2otf([1;-1],[N N]);
D.DhT_DFT_b = conj(D.Dh_DFT_b);
D.DvT_DFT_b = conj(D.Dv_DFT_b);

%% Intenisty and background estimation

ynew = mean(Y_nobound,3);

% options for the intensity estimation
col0rme2.est_var = var_noise;

% General Initializaitons
col0rme2.alpha = 1e6;   % the regularization parameter alpha (chozen big) 
col0rme2.maxInIt = 1e3;
col0rme2.maxInIt_xb = 50;
col0rme2.tol = 1e-6;

col0rme2.LipD = max(max(abs(D.DhT_DFT .* D.Dh_DFT + D.DvT_DFT .* D.Dv_DFT)));
col0rme2.LipA = max(norm(Mech*MechT)*abs(PSFfft(:)))^2;
col0rme2.LipI = 1;
col0rme2.LipDb = max(max(abs(D.DhT_DFT_b .* D.Dh_DFT_b + D.DvT_DFT_b .* D.Dv_DFT_b)));

col0rme2.pixels_fg = pixels_fg;
col0rme2.pixels_cg = pixels_cg;

col0rme2.maxIt = 1e4;
col0rme2.mu_init = 0.1; % the initialization of the regularization parameter for intensity
col0rme2.tol_f = 1e-3;

col0rme2.v_DP =sqrt(opts.K); % \nu_{DP}, the parameter of discrepancy princi
col0rme2.lambda = 100;       % the regularization parameter for the background

if ~isfile([opts.folderSave 'COL0RME_step2_K' num2str(opts.K) '_N' num2str(N) '.mat'])
      
    sol_col0rme2 = DiscrepancyPrinciple(col0rme2, ynew, PSFfft, D, Mech, is, opts);
%     sol_col0rme2 = DiscrepancyPrincipleNorm(col0rme2, ynew, PSFfft, D, Mech, is, opts); % Normalize every norm with the number of elements

    save([opts.folderSave 'COL0RME_step2_K' num2str(opts.K) '_N' num2str(N)], 'col0rme2', 'sol_col0rme2');
else 
    load([opts.folderSave 'COL0RME_step2_K' num2str(opts.K) '_N' num2str(N) '.mat'])
end

%% Grid search

% Grid search
range = 0.01:0.02:0.21;
[f_mu_range, psnr_range] = grid_search(col0rme2, sol_col0rme2, range, ynew, PSFfft, D, Mech, is, opts);
% [f_mu_range, psnr_range] = grid_searchNorm(col0rme2, sol_col0rme2, range, ynew, PSFfft, D, Mech, is, opts); % Normalize every norm with the number of elements, goes with: DiscrepancyPrincipleNorm 
