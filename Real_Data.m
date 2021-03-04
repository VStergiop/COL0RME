clear all;
close all;
clc;

addpath algorithms 
addpath util

%% Evaluation parameters
opts.K = 1000;              % Stack lenght
opts.GPU = 1;               % 0: CPU, 1: GPU
opts.BackEst = 2;           % 0: without Background Estimation, 
                            % 1: with B.E., Back: constant in space
                            % 2: with B.E., Back: varying in space
opts.compute = false;       % compute f/mu/b/psnr for Discrepancy Principle throuth the iterations
opts.sim=0;                 % for the computation of the GT image, PSNR value
opts.patches = 1;           % 0: without patches, 1: with patches 

opts.folderSave = 'Results_RealData';     % The folder to save the result
opts.indY = {80+(1:172), 340+(1:172)};  % 1 patch:40, 2:62, 3:84, 4:106, 5:128, 6:150, 7:172   

save([opts.folderSave 'parameters_K' num2str(opts.K) '.mat'],'opts')

%% Data
filename = 'angle2.tiff';
y = imread(filename);

Ytotal = zeros(size(y,1),size(y,2),opts.K);
k0=0;
for k = 1:opts.K
    Ytotal(:,:,k) = imread(filename, k0 + k);
end 

Y = Ytotal(opts.indY{1}, opts.indY{2},:);

%% Parmaters for dictionary
pixsiz = 106; % Pixel size in nm
lamb = 525;  % Emission Wavelenght in nm
NA = 1.33; % Numerical aperture
fwhm = 0.61 * lamb/NA ;  % Width of PSF in nm
srf = 4; % Super-resolution factor

%% COL0RME

% Affected pixel
sigma = fwhm / (2*sqrt(2*log(2)));
pixels_cg = round(3 * sigma / pixsiz);  % pixels in coarse grid
pixels_fg = srf*pixels_cg;      % pixels in fine grid

% Support and Noise Variance
if opts.patches 
    % Parameters 
    bound_cg = 3*pixels_cg; % fix the boundaries I am going to remove
    size_patch = 40;        % size of the patch
    
    % Initializations
    num_patch = (numel(opts.indY{1}) - 2*bound_cg)/(size_patch-2*bound_cg); % number of patches
    supp_x = zeros(num_patch*srf*(size_patch-2*bound_cg)); % support image
    patch_var_noise = zeros(num_patch); % noise varianced computed for each patch
    patch_noSign = zeros(num_patch); % check in which patches we haven't signal (gaussian noise and not poisson)
    
    for i=1:num_patch
        for j = 1:num_patch
            Ynoisy = Y((i-1)*(size_patch-2*bound_cg)+1:(i-1)*(size_patch-2*bound_cg)+size_patch,(j-1)*(size_patch-2*bound_cg)+1:(j-1)*(size_patch-2*bound_cg)+size_patch,:);
            run('COL0RME_support')
            if nnz(sol_col0rme.suppX(bound_cg*srf+1:end-bound_cg*srf,bound_cg*srf+1:end-bound_cg*srf))<10
                patch_noSign(i,j)=1;
            end
            supp_x((i-1)*srf*(size_patch-2*bound_cg)+1:i*srf*(size_patch-2*bound_cg),(j-1)*srf*(size_patch-2*bound_cg)+1:j*srf*(size_patch-2*bound_cg))=...
                sol_col0rme.suppX(bound_cg*srf+1:end-bound_cg*srf,bound_cg*srf+1:end-bound_cg*srf);
            patch_var_noise(i,j) = double(gather(sol_col0rme.VarNoise)) * (max(Ynoisy(:)))^2;
        end
    end
    var_noise = mean(patch_var_noise(patch_noSign==1));
else
    % Parameters 
    bound_cg = pixels_cg; % fix the boundaries I am going to remove
    
    Ynoisy = Y;
    run('COL0RME_support')
    supp_x = sol_col0rme.suppX(bound_cg*srf+1:end-bound_cg*srf,bound_cg*srf+1:end-bound_cg*srf);
    var_noise = double(gather(sol_col0rme.VarNoise)) * (max(Ynoisy(:)))^2;
end
    
% Intnsity and Background
Y_nobound = Y(bound_cg+1:end-bound_cg,bound_cg+1:end-bound_cg,:);
run('COL0RME_intensity')

% Save and Plot

figure
subplot 221
imagesc(mean(Y_nobound(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg,:),3))
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title('Mean of the stack')
subplot 222
imagesc(supp_x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg))
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title('Estimated support')
subplot 223
imagesc(sol_DiscPrinc.x(pixels_fg+1:end-pixels_fg,pixels_fg+1:end-pixels_fg));
colormap(gca,'hot')
pbaspect([1 1 1])
colorbar   
title('Intensity')
if opts.BackEst == 2
    subplot 224
    imagesc(sol_DiscPrinc.b(pixels_cg+1:end-pixels_cg,pixels_cg+1:end-pixels_cg));
    colormap(gca,'hot')
    pbaspect([1 1 1])
    colorbar   
    title(['Background, \lambda = ' num2str(col0rme.optsIn.lambda) ])
end