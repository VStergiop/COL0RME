%% Build dictionary

% Coarse grid size
N = size(Ynoisy,1); 

% Super-resolution grid size
L = N * srf;

% Build dictionary
[lambdah, Gl, Sm] = gausspsfcirc(N, pixsiz, srf, fwhm, false);
GlGl = khatrirao(Gl,Gl);

% Dictionary
[~,Gh] = gausspsfcirc(L, pixsiz/srf, 1, fwhm, false);
Gh0 = Gh(:,1) * Gh(:,1)';

% Auxiliary variables, for efficiency
replambdah = repmat(lambdah(:), [1 L]);
replambdah = replambdah .* (replambdah.');

% Efficient subsampling (integration) operator
SmSm = @(x) squeeze(sum(reshape(squeeze(sum(reshape(x, [L srf N]), 2))', [N srf N]), 2))';

% Computing Lipschitz constant 
try
    load([opts.folderSave 'Lipschitz_tubulin_N' num2str(N) '.mat'])
catch
    if opts.GPU
        [Lip, ~] = pwritr_GPU(randn(L^2,1), Gl, 500, 1e-3);
        Lip = gather(Lip);
    else
        [Lip, ~] = pwritr(randn(L^2,1), Gh0, GlGl, replambdah, SmSm, 500, 1e-3);
    end
    save([opts.folderSave 'Lipschitz_tubulin_N' num2str(N) '.mat'],'Lip')
end

%% Correlation CEL0 method -- algorithm options

col0rme.opts.maxit = 5e2;
col0rme.opts.tol = 2e-5;
col0rme.opts.dispfac = 1;
col0rme.opts.verbose = true;

% FISTA options
col0rme.opts.opts_fista.maxit = 2e4;
col0rme.opts.opts_fista.tol = 2e-5;
col0rme.opts.opts_fista.verbose = true;
col0rme.opts.opts_fista.dispfac = 1;
col0rme.opts.opts_fista.computef = false;

% Computing norms for later use in computation of regulaization param
col0rme.normsG = sum(Gl.^2);
col0rme.normsG = kron(col0rme.normsG, col0rme.normsG);
col0rme.normsG = col0rme.normsG(:);

% Support estimation parameter (regularization)
col0rme.gamma =  1e-9;

%% COL0RME

if opts.patches 
    patch = ['_' num2str(i) num2str(j)];
else
    patch = '';
end

if ~isfile([opts.folderSave 'COL0RME_step1_K' num2str(opts.K) '_N' num2str(N) patch '.mat'])

    % Computing covariance matrix
    Ry = cov(munfold(Ynoisy/max(Ynoisy(:)),3,[2 1]));

    % Computing reference regularization param
    col0rme.gamma0 = reshape(Ry(:), [N N N N]);
    col0rme.gamma0 = munfold(col0rme.gamma0, [1 3], [2 4]);
    col0rme.gamma0 = GlGl' * col0rme.gamma0 * GlGl;
    col0rme.gamma0 = col0rme.gamma0(:) ./ sqrt(2*col0rme.normsG(:).^2);
    col0rme.gamma0 = max(col0rme.gamma0(:));

    % Applying algorithm
    exttimer = tic;
    if opts.GPU
        [sol_col0rme.Rx, sol_col0rme.VarNoise, sol_col0rme.outinfo] = corrcelo_VarN_GPU(Ry(:), Gl, ...
                        replambdah, Gh0, SmSm, Lip, srf, col0rme.gamma * col0rme.gamma0 * ones(L^2,1), col0rme.opts);
        sol_col0rme.Rx = gather(sol_col0rme.Rx);
    else
        [sol_col0rme.Rx, sol_col0rme.VarNoise, sol_col0rme.outinfo] = corrcelo_VarN_CPU(Ry(:), Gl, ...
                        replambdah, Gh0, SmSm, Lip, srf, col0rme.gamma * col0rme.gamma0 * ones(L^2,1) , col0rme.opts);
    end
    sol_col0rme.comptime = toc(exttimer);

     % Extracting estimated support
    sol_col0rme.suppX = sol_col0rme.Rx;
    sol_col0rme.suppX(sol_col0rme.suppX > 0) = 1;
    sol_col0rme.isuppX = find(sol_col0rme.suppX(:) > 0);     

    % Image format
    sol_col0rme.suppX = reshape(sol_col0rme.suppX, [L L]);
    sol_col0rme.Rx = reshape(sol_col0rme.Rx, [L L]);

    % Storing results
    save([opts.folderSave 'COL0RME_step1_K' num2str(opts.K) '_N' num2str(N) patch], 'col0rme', 'sol_col0rme');
else 
    load([opts.folderSave 'COL0RME_step1_K' num2str(opts.K) '_N' num2str(N) patch '.mat'])
end