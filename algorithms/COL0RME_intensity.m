function sol_col0rme_int = COL0RME_intensity(Y_nobound, supp_x, var_noise,opts) 
    
    % Coarse grid size
    N = size(Y_nobound,1); 

    % Super-resolution grid size
    L = N * opts.srf;

    % Build dictionary
    PSF = gausspsf(N, opts.pixsiz, opts.srf, opts.fwhm);

    PSFfft = fft2(fftshift(PSF));

    Mech = zeros(N, L);
    for i = 0 : N-1
        Mech(i+1, 1+opts.srf*i : opts.srf+opts.srf*i) = 1;
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
    col0rme_int.est_var = var_noise;

    % General Initializaitons
    col0rme_int.alpha = 1e6;   % the regularization parameter alpha (chozen big) 
    col0rme_int.maxInIt = 1e3;
    col0rme_int.maxInIt_xb = 50;
    col0rme_int.tol = 1e-6;

    col0rme_int.LipD = max(max(abs(D.DhT_DFT .* D.Dh_DFT + D.DvT_DFT .* D.Dv_DFT)));
    col0rme_int.LipA = max(norm(Mech*MechT)*abs(PSFfft(:)))^2;
    col0rme_int.LipI = 1;
    col0rme_int.LipDb = max(max(abs(D.DhT_DFT_b .* D.Dh_DFT_b + D.DvT_DFT_b .* D.Dv_DFT_b)));

%     col0rme_int.pixels_fg = pixels_fg;
%     col0rme_int.pixels_cg = pixels_cg;

    col0rme_int.maxIt = 2e4;
    col0rme_int.mu_init = 0.1; % the initialization of the regularization parameter for intensity
    col0rme_int.tol_f = 1e-2;

    col0rme_int.v_DP =sqrt(opts.K);                 % \nu_{DP}, the parameter of discrepancy principle
    col0rme_int.lambda = opts.regpar_back;          % the regularization parameter for the background

    if ~isfile([opts.folderSave 'COL0RME_int_tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '_' opts.reg '.mat'])

        sol_col0rme_int = DiscrepancyPrinciple(col0rme_int, ynew, PSFfft, D, Mech, is, opts);

        save([opts.folderSave 'COL0RME_int_tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '_' opts.reg '.mat'], 'col0rme_int', 'sol_col0rme_int');
    else 
        load([opts.folderSave 'COL0RME_int_tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '_' opts.reg '.mat'])
    end

    %% Grid search

    % Grid search
    % range = 0.2:0.05:0.3;
    % [f_mu_range, psnr_range] = grid_search(col0rme2, sol_col0rme2, range, ynew, PSFfft, D, Mech, is, opts);
    % [f_mu_range, psnr_range] = grid_searchNorm(col0rme2, sol_col0rme2, range, ynew, PSFfft, D, Mech, is, opts); % Normalize every norm with the number of elements, goes with: DiscrepancyPrincipleNorm 
end