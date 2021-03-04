clear all;
close all;
clc;

addpath algorithms 
addpath util

%% Evaluation parameters
opts.K = 500;               % Stack lenght
opts.SNR = 12;              % Measurement noise SNR
opts.GPU = 1;               % 0: CPU, 1: GPU, at the moment only CPU for 2nd step
opts.BackEst = 1;           % 0: without Background Estimation, 
                            % 1: with B.E., Back: constant in space
                            % 2: with B.E., Back: varying in space
opts.compute = true;        % compute f/mu/b/psnr evolution for Discrepancy Principle
opts.sim = 1;               % simulated data, compute GT image and PSNR values
opts.patches = 0;           % 0: without patches, only this option for sim. data

opts.folderSave = 'Results_SimData/';   % The folder to save the result
opts.filename = 'tubulin_noiseless_highBg.mat'; % The filename 
opts.indsY = {12+(1:24), 12+(1:24)}; % the indices, in raw data

save([opts.folderSave 'parameters_K' num2str(opts.K) '.mat'],'opts')
%% Data

% Load data
load(opts.filename);
Y = double(stacks_discrete(opts.indsY{1}, opts.indsY{2}, :));
clear stacks_discrete

% Image dimensions
N = size(Y,1);

%% Add Gaussian noise 
if ~isfile([opts.folderSave 'tubulin_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '.mat'])

    Yn = Y(:, :, 1:opts.K);
    
    % Noise tensor
    Noise = randn(size(Yn));
    sigma_noise = sqrt(var(Yn(:))/10^(opts.SNR/10)); % !!Is not exactly the SNR I have defined before!!
    
    % Noisy stack
    Ynoisy = Yn + sigma_noise * Noise ;
    % Ynoisy(Ynoisy<0)=0;
    
    % Saving used data
    save([opts.folderSave 'tubulin_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N)], 'Yn', 'Ynoisy', 'sigma_noise', 'Noise');
    savestack([opts.folderSave 'tubulin_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '.tiff'], uint16(Ynoisy )); 
else
    load([opts.folderSave 'tubulin_SNR' num2str(opts.SNR) '_K' num2str(opts.K) '_N' num2str(N) '.mat']);
end

%% Parmaters for dictionary
pixsiz = Cam.pixel_size * 1e9;                          % Pixel size in nm
fwhm = Cam.pixel_size * Optics.fwhm_digital * 1e9;      % Width of PSF in nm
srf = 4;                                                % Super-resolution factor

%% CAL0RME
% Affected pixel
sigma = fwhm / (2*sqrt(2*log(2)));
pixels_cg = round(3 * sigma / pixsiz);  % pixels in coarse grid
pixels_fg = srf*pixels_cg;              % pixels in fine grid

% Ground Truth image
ground_truth =  GT_image(Fluo, Cam, Grid, srf, opts.indsY, opts.K);
x_GT = ground_truth.int_K;
opts.x_GT = x_GT(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);              % remove the boundaries from the GT image

% Support and Noise Variance
run('COL0RME_support')
supp_x = sol_col0rme.suppX(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg);    % remove the boundaries from the support image
var_noise = double(gather(sol_col0rme.VarNoise)) * (max(Ynoisy(:)))^2;              % variance of the noise in the correct scale           

% Intnsity and Background
Y_nobound = Ynoisy(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:);          % remove the boundaries from the stack
run('COL0RME_intensity')

% Save and Plot

figure
subplot 231
imagesc(mean(Y_nobound(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:),3))
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title('Mean of the stack')
subplot 232
imagesc(opts.x_GT(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg))
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title('x_{GT}')
subplot 233
imagesc(supp_x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg))
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title('Estimated support')
subplot 234
imagesc(sol_col0rme2.x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg));
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title(['Intensity, PSNR:' num2str(sol_col0rme2.psnr) 'dB'] )
if opts.BackEst == 2
    subplot 235
    imagesc(sol_col0rme2.b(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg));
    colormap(gca,'hot')
    pbaspect([1 1 1])
    colorbar   
    title(['Background, \lambda = ' num2str(col0rme2.lambda) ])
end
