clear all;
close all;
clc;

addpath algorithms 
addpath util


%% OPTIONS

% Evaluation parameters
opts.K = 500;               % Stack lenght
opts.SNR = 15;              % Measurement noise SNR
opts.GPU = 1;               % 0: CPU, 1: GPU, at the moment only CPU for 2nd step
opts.reg = 'L1';            % other choises: 'CEL0_0', 'L1'
opts.BackEst = 2;           % 0: without Background Estimation(B.E), 
                            % 1: with B.E., Back: constant in space
                            % 2: with B.E., Back: varying in space
opts.VarEst = 1;            % 0: without Variance Estimation - ONLY to prove how important is the VarEst: GPU&(CEL0_0/L1)
                            % 1: with Variance Estimation
opts.compute = false;        % compute f/mu/b/psnr evolution for Discrepancy Principle
opts.patches = 0;           % no patches implementation for simulated data

% Regularization parameters
opts.regpar_supp = 5e-4;        %regularization parameter -- support
opts.regpar_back = 20;          %regularization parameter -- background

% File + Saving Folder
opts.folderSave = 'Results/';                           % The folder to save the result
opts.ref = 'highBg';                                    % other background choises: 'noBg', 'lowBg', highBg'
filename = ['tubulin_noiseless_' opts.ref '.mat'];      % the dataset 
opts.indsY = {12+(1:40), 12+(1:40)};                    % indices for cutting the frames

% Restarting
if strcmp(opts.reg,'CEL0_0')
    opts.iter = 10;
else
    opts.iter = 1;
end


%% Data
% Load data
load(filename);
Y = double(stacks_discrete(opts.indsY{1}, opts.indsY{2}, :));
clear stacks_discrete

% Image dimensions
N = size(Y,1);

% Add Gaussian noise 
if ~isfile([opts.folderSave 'tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N) '.mat'])

    Yn = Y(:, :, 1:opts.K);

    % Noise tensor
    Noise = randn(size(Yn));
    sigma_noise = sqrt(var(Yn(:))/10^(opts.SNR/10)); 

    %Noisy stack
    Ynoisy = Yn + sigma_noise * Noise ;

    % Saving used data
    save([opts.folderSave 'tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N)], 'Yn', 'Ynoisy', 'sigma_noise', 'Noise');
    savestack([opts.folderSave 'tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N) '.tiff'], uint16(Ynoisy)); 
    clear Yn Noise 
else
    load([opts.folderSave 'tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N) '.mat'],'Ynoisy', 'sigma_noise');
end
clear Y

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
clear Fluo Cam Grid Optics
x_GT = ground_truth.int_K;
opts.x_GT = x_GT(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);                           % remove the boundaries from the GT image    

%% COL0RME: Support estimation

if ~isfile([opts.folderSave 'COL0RME_supp_tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N) '.mat'])

    supp_sp = zeros(N*opts.srf-2*pixels_fg);        % support superposition
    var_noise_mat = zeros(opts.iter,1);             % var_noise_matrix

    nnz_elem_p=0;
    k=1;
    while k<=opts.iter

        opts.init_X  = zeros(N*opts.srf);
        if k>1
            % INITIALIZE
            pos_nze = cell2mat(myind2sub([N*opts.srf N*opts.srf],find(sol_col0rme_supp.Rx > 0)));
            new_loc = det_init(pos_nze(:,1)'-0.5, pos_nze(:,2)'-0.5);
            % new_loc = stoc_init(pos_nze(:,1)'-0.5, pos_nze(:,2)'-0.5, opts.srf*round(sigma_gauss / opts.pixsiz));

            for j = 1:size(new_loc,1)
                if new_loc(j,1)>0 && new_loc(j,2)>0 && new_loc(j,1)<=N*opts.srf && new_loc(j,2)<=N*opts.srf 
                    opts.init_X (new_loc(j,1),new_loc(j,2))=1;
                end
            end
        end

        opts.init_X = opts.init_X(:);

        % Support and Noise Variance
        sol_col0rme_supp = COL0RME_support(Ynoisy,opts);
        supp_x = sol_col0rme_supp.suppX(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);       % remove the boundaries from the support image
        var_noise_mat(k) = sol_col0rme_supp.VarNoise;   
        supp_sp = supp_sp+supp_x; supp_sp(supp_sp>0)=1;

        nnz_elem = nnz(supp_sp);
        if nnz_elem == nnz_elem_p
            break
        end
        nnz_elem_p = nnz_elem;
        k=k+1;
    end
    save([opts.folderSave 'COL0RME_supp_tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N)], 'supp_sp','var_noise_mat', 'sol_col0rme_supp', 'opts');
    clear sol_col0rme_supp
else 
    load([opts.folderSave 'COL0RME_supp_tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(N) '.mat'], 'supp_sp','var_noise_mat')
end

% Noise Variance
iter =  nnz(var_noise_mat);
var_noise = sum(var_noise_mat(:))/iter;

% Jaccard Index
[Jacc_col0rme_Diag,Jacc_col0rme_Ver] = JI(opts.x_GT, supp_sp, opts.pixsiz, opts.srf);
fprintf('JI: %4.1e\n',  Jacc_col0rme_Diag.all);


%% COL0RME: Intensity and Background estimation

Y_nobound = Ynoisy(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:);          % remove the boundaries from the stack

if ~isfile([opts.folderSave 'COL0RME_int_tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(size(Y_nobound,1)) '.mat'])
    
    [col0rme_int, sol_col0rme_int] = COL0RME_intensity(Y_nobound, supp_sp, var_noise, opts);

    save([opts.folderSave 'COL0RME_int_tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(size(Y_nobound,1)) '.mat'], 'col0rme_int', 'sol_col0rme_int', 'opts');
    clear col0rme_int
else 
    load([opts.folderSave 'COL0RME_int_tubulin_' opts.ref '_K' num2str(opts.K) '_N' num2str(size(Y_nobound,1)) '.mat'],'sol_col0rme_int')
end

% PSNR 
int_x = sol_col0rme_int.x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);
x_GT_cut = opts.x_GT(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);
psnr_col0rme = computePSNR(x_GT_cut,int_x);
fprintf('PSNR: %4.1e\n', psnr_col0rme);


% Plot the results
Y_cut = mean(Y_nobound(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:),3);
supp_x_cut = supp_sp(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg); 
b = sol_col0rme_int.b(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg);
plot_results_COL0RME(Y_cut,supp_x_cut,int_x,b,opts,x_GT_cut,Jacc_col0rme_Diag.all,psnr_col0rme,sol_col0rme_int.DP,var_noise,sigma_noise^2)      

