clear all;
close all;
clc;

addpath algorithms 
addpath util

for K_stack = 700%[700 100]
    for SNR_stack = 10%[20 10]
        
        %% OPTIONS
        % Evaluation parameters
        opts.K = K_stack;           % Stack lenght
        opts.SNR = SNR_stack;       % Measurement noise SNR
        opts.GPU = 1;               % 0: CPU, 1: GPU, at the moment only CPU for 2nd step
        opts.reg = 'L1';        % 'CEL0_0', 'CEL0_l1', 'L1', 'TV'
        opts.BackEst = 2;           % 0: without Background Estimation(B.E), 
                                    % 1: with B.E., Back: constant in space
                                    % 2: with B.E., Back: varying in space
        opts.VarEst = 1;            % 0: without Variance Estimation - ONLY to prove how important is the VarEst: GPU&(CEL0_0/L1)
                                    % 1: with Variance Estimation
        opts.compute = true;        % compute f/mu/b/psnr evolution for Discrepancy Principle
        opts.patches = 0;           % no patches implementation for simulated data
        
        % Regularization parameters
        opts.regpar_supp = 5e-7;        %regularization parameter -- support
        opts.regpar_back = 20;          %regularization parameter -- background
        
        % File + Saving Folder
        opts.folderSave = 'Simulated_Data/';                % The folder to save the result
        opts.bg = 'lowBg';                                  % lowBg, highBg, noBg
        filename = ['tubulin_noiseless_' opts.bg '.mat'];   % The filename 
        opts.indsY = {12+(1:40), 12+(1:40)};                % the indices, in raw data
        
       
        %% Data
        % Load data
        load(filename);
        Y = double(stacks_discrete(opts.indsY{1}, opts.indsY{2}, :));
        clear stacks_discrete

        % Image dimensions
        N = size(Y,1);

        % Add Gaussian noise 
        if ~isfile([opts.folderSave 'tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '.mat'])

            Yn = Y(:, :, 1:opts.K);

            % Noise tensor
            Noise = randn(size(Yn));
            sigma_noise = sqrt(var(Yn(:))/10^(opts.SNR/10)); % !!Is not exactly the SNR I have defined before!!

            %Noisy stack
            Ynoisy = Yn + sigma_noise * Noise ;
            % Ynoisy(Ynoisy<0)=0;

            % Saving used data
            save([opts.folderSave 'tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N)], 'Yn', 'Ynoisy', 'sigma_noise', 'Noise');
            savestack([opts.folderSave 'tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '.tiff'], uint16(Ynoisy )); 
        else
            load([opts.folderSave 'tubulin_' opts.bg '_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '.mat']);
        end

        % Parmaters for dictionary
        opts.pixsiz = Cam.pixel_size * 1e9;                          % Pixel size in nm
        opts.fwhm = Cam.pixel_size * Optics.fwhm_digital * 1e9 ;     % Width of PSF in nm
        opts.srf = 4;                                                % Super-resolution factor

        % Affected pixel
        sigma_gauss = opts.fwhm / (2*sqrt(2*log(2)));
        pixels_cg = round(3 * sigma_gauss / opts.pixsiz);           % pixels in coarse grid
        pixels_fg = opts.srf*pixels_cg;                             % pixels in fine grid

        % Ground Truth image
        ground_truth =  GT_image(Fluo, Cam, Grid, opts.srf, opts.indsY, opts.K);
        x_GT = ground_truth.int_K;
        
        %% COL0RME

        % Support and Noise Variance
        sol_col0rme_supp = COL0RME_support(Ynoisy,opts);
        var_noise = sol_col0rme_supp.VarNoise;               
        fprintf('The estimated variance of the noise: %4.1e and the real: %4.1e\n',var_noise,sigma_noise^2)
        
        %Jaccard Index
        supp_x = sol_col0rme_supp.suppX(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);       % remove the boundaries from the support image
        x_GT = x_GT(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);                           % remove the boundaries from the GT image    
        [Jacc_col0rme_Diag,Jacc_col0rme_Ver] = JI(x_GT,supp_x, opts.pixsiz, opts.srf);
 
        % Intensity and Background
        Y_nobound = Ynoisy(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:);          % remove the boundaries from the stack
        sol_col0rme_int = COL0RME_intensity(Y_nobound, supp_x, var_noise, opts);
        
        % PSNR 
        int_x = sol_col0rme_int.x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);
        x_GT_cut = x_GT(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);
        psnr_col0rme = computePSNR(x_GT_cut,int_x);
        
        % Plot the results
        
        Y_cut = mean(Y_nobound(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:),3);
        supp_x_cut = supp_x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg); 
        b = sol_col0rme_int.b(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg);
        plot_results_COL0RME(Y_cut,supp_x_cut,int_x,b,x_GT_cut,Jacc_col0rme_Diag.all,psnr_col0rme,opts)      
    end
end

 