function sol_col0rme_supp = COL0RME_support(Ynoisy,opts)
    
    %% Build dictionary

    % Coarse grid size
    N = size(Ynoisy,1); 

    % Super-resolution grid size
    L = N * opts.srf;

    % Build dictionary
    [lambdah, Gl, ~] = gausspsfcirc(N, opts.pixsiz, opts.srf, opts.fwhm, false);
    GlGl = khatrirao(Gl,Gl);

    % Dictionary
    [~,Gh] = gausspsfcirc(L, opts.pixsiz/opts.srf, 1, opts.fwhm, false);
    Gh0 = Gh(:,1) * Gh(:,1)';

    % Auxiliary variables, for efficiency
    replambdah = repmat(lambdah(:), [1 L]);
    replambdah = replambdah .* (replambdah.');

    % Efficient subsampling (integration) operator
    SmSm = @(x) squeeze(sum(reshape(squeeze(sum(reshape(x, [L opts.srf N]), 2))', [N opts.srf N]), 2))';

    % Computing Lipschitz constant 
    try
        load([opts.folderSave 'Lipschitz_tubulin_' opts.ref '_N' num2str(N) '.mat'])
    catch
        if opts.GPU
            [Lip, ~] = pwritr_GPU(randn(L^2,1), Gl, 500, 1e-3);
            Lip = gather(Lip);
        else
            [Lip, ~] = pwritr(randn(L^2,1), Gh0, GlGl, replambdah, SmSm, 500, 1e-3);
        end
        save([opts.folderSave 'Lipschitz_tubulin_' opts.ref '_N' num2str(N) '.mat'],'Lip')
    end

    %% COL0RME SUPPORT -- algorithm options

    col0rme_supp.opts.maxit = 5e2;
    col0rme_supp.opts.tol = 2e-5;
    col0rme_supp.opts.dispfac = 1;
    col0rme_supp.opts.verbose = true;

    % FISTA options
    col0rme_supp.opts.opts_fista.maxit = 2e4;
    col0rme_supp.opts.opts_fista.tol = 2e-5;
    col0rme_supp.opts.opts_fista.verbose = true;
    col0rme_supp.opts.opts_fista.dispfac = 1;
    col0rme_supp.opts.opts_fista.computef = false;

    % Computing norms for later use in computation of regulaization param
    col0rme_supp.normsG = sum(Gl.^2);
    col0rme_supp.normsG = kron(col0rme_supp.normsG, col0rme_supp.normsG);
    col0rme_supp.normsG = col0rme_supp.normsG(:);

    % Support estimation parameter (regularization)
    col0rme_supp.gamma =  opts.regpar_supp;

    % initialization
    col0rme_supp.opts.X0 = opts.init_X;
    
    %% COL0RME SUPPORT

    % Computing covariance matrix
    Ry = cov(munfold(Ynoisy,3,[2 1]));
    norm_factor = max(Ry(:));
    Ry = Ry/norm_factor;

    % Computing reference regularization parameter for CEL0
    col0rme_supp.gamma0 = reshape(Ry(:), [N N N N]);
    col0rme_supp.gamma0 = munfold(col0rme_supp.gamma0, [1 3], [2 4]);
    col0rme_supp.gamma0 = GlGl' * col0rme_supp.gamma0 * GlGl;
    col0rme_supp.gamma0_cel0 = (col0rme_supp.gamma0(:).^2)./(2*col0rme_supp.normsG(:).^2); 
    col0rme_supp.gamma0_cel0 = max(col0rme_supp.gamma0_cel0(:));
    col0rme_supp.gamma0_l1 = max(col0rme_supp.gamma0(:));

    % Applying algorithm
    exttimer = tic;
    switch opts.reg
        case 'CEL0_0'
            if opts.GPU
               if opts.VarEst
                    [sol_col0rme_supp.Rx, sol_col0rme_supp.VarNoise, sol_col0rme_supp.outinfo] = corrCEL0_VarN_GPU(Ry(:), Gl, replambdah, ...
                                    Gh0, SmSm, Lip, opts.srf, col0rme_supp.gamma * col0rme_supp.gamma0_cel0 * ones(L^2,1), col0rme_supp.opts);
                    sol_col0rme_supp.Rx = gather(sol_col0rme_supp.Rx);
               else
                   [sol_col0rme_supp.Rx, sol_col0rme_supp.outinfo] = corrCEL0_noVarN_GPU(Ry(:), Gl, replambdah, ...
                                    Gh0, SmSm, Lip, opts.srf, col0rme_supp.gamma * col0rme_supp.gamma0_cel0 * ones(L^2,1), col0rme_supp.opts);
                    sol_col0rme_supp.Rx = gather(sol_col0rme_supp.Rx);
               end
            else
                [sol_col0rme_supp.Rx, sol_col0rme_supp.VarNoise, sol_col0rme_supp.outinfo] = corrCEL0_VarN_CPU(Ry(:), Gl, replambdah, ...
                                Gh0, SmSm, Lip, opts.srf, col0rme_supp.gamma * col0rme_supp.gamma0_cel0 * ones(L^2,1) , col0rme_supp.opts);
            end
        case 'L1'
            if opts.GPU
                if opts.VarEst
                    [sol_col0rme_supp.Rx, sol_col0rme_supp.VarNoise] = corrL1_VarN_GPU(Ry(:), Gl, Lip, opts.srf,...
                                    col0rme_supp.gamma * col0rme_supp.gamma0_cel0 , col0rme_supp.opts);
                    sol_col0rme_supp.Rx = gather(sol_col0rme_supp.Rx); 
                else
                    [sol_col0rme_supp.Rx] = corrL1_noVarN_GPU(Ry(:), Gl, Lip, opts.srf,...
                                    col0rme_supp.gamma * col0rme_supp.gamma0_l1 , col0rme_supp.opts);
                    sol_col0rme_supp.Rx = gather(sol_col0rme_supp.Rx); 
                end
            else
                error('Only GPU implementation. Try CEL0_0 regularizer for CPU implementation');
            end
    end    
    sol_col0rme_supp.comptime = toc(exttimer);

    % Extracting estimated support
    sol_col0rme_supp.suppX = sol_col0rme_supp.Rx;
    sol_col0rme_supp.suppX(sol_col0rme_supp.suppX > 0) = 1;
    sol_col0rme_supp.isuppX = find(sol_col0rme_supp.suppX(:) > 0);     

    % Image format
    sol_col0rme_supp.suppX = reshape(sol_col0rme_supp.suppX, [L L]);
    sol_col0rme_supp.Rx = reshape(sol_col0rme_supp.Rx, [L L]);

    if opts.VarEst
        % rescale the Variance of the noise
        sol_col0rme_supp.VarNoise = double(gather(sol_col0rme_supp.VarNoise))*norm_factor;
    end
        