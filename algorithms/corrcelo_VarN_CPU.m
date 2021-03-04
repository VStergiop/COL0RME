function [X, Var_noise, outinfo] = corrcelo_VarN_CPU(Y, Gl, replambdah, Gh0, SmSm, Lip, q, gamma, opts)

% Display options
defopts.verbose = true;     % Verbose mode
defopts.dispfac = 1;        % Displays info only at multiples of this number
defopts.computef = true;

% Stopping criteria
defopts.maxit = 1e3;           % Maximum number of iterations
defopts.tol = 1e-4;            % Tolerance for declaring convergence
defopts.checkconv = true;      % Whether to stop algorithm if conv. is detected

% FISTA parameters
defopts.opts_fista.maxit = 1e3;           % Maximum number of iterations
defopts.opts_fista.tol = 1e-4;            % Tolerance for declaring convergence
defopts.opts_fista.verbose = true;     % Verbose mode
defopts.opts_fista.dispfac = 1;        % Displays info only at multiples of this number
defopts.opts_fista.computef = false;

% Background pattern
defopts.b = eye(numel(Y)^(1/2))/numel(Y)^(1/2);

% Initial solution
defopts.X0 = [];
defopts.Xpos = [];
defopts.sigb0 = [];


% Obtaning parameters
if nargin < 9 || isempty(opts)
    opts = defopts;
else                       % copies only undefined params
    opts = completestruct(opts,defopts);
    opts.opts_fista = completestruct(opts.opts_fista,defopts.opts_fista);
    opts.opts_fista.verbose = opts.verbose;        
end

%%%  Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(Y,1)^(1/4);
L = N * q;
L2 = L*L;

GlGl = khatrirao(Gl,Gl);

% Pre-computing contants needed for updating weights
normsG = sum(Gl.^2); 
normsG = [kron(normsG, normsG)];
normsG2 = normsG(:).^2;

sq2g_div_nG   = sqrt(2*gamma) ./ normsG(:);

sq2g_times_nG = sqrt(2*gamma) .* normsG(:);

if ~isempty(opts.sigb0)
    sigb = opts.sigb0;
else
    sigb = 0;
end

% Initial solution
if ~isempty(opts.X0)
    X = opts.X0;
elseif ~isempty(opts.Xpos)
    X = zeros(L2, 1);
    X(opts.Xpos, :) = rand(length(opts.Xpos), 1);
else
    if opts.verbose    
        disp('Initializing with FISTA solution (convex formulation) ...')
    end
    X = zeros(L^2, 1);
    [X, sigb, ~] = wl1fista(Y, GlGl, replambdah, Gh0, SmSm, X, sigb, opts.b, gamma, Lip, opts.opts_fista);
    if opts.verbose    
        fval = costfncelo(Y, X, sigb, opts.b, Gh0, replambdah, SmSm, gamma, sq2g_div_nG, normsG2, abs([X(:)]) < sq2g_div_nG(:));
        fprintf('Computed initial point, cost fn value = %5.2e \n',fval);
    end
end

% Display header
if opts.verbose
    hline = '=============================================================================';
    header            = '   it          relerr     inn. it    cost     err     W|X|_{2,1}    pixels   (grid)';
    subheader = sprintf('(max=%4d)  (tol=%4.1e)', opts.maxit, opts.tol);
end

% Preparing outputs
if opts.computef
    outinfo.fval = zeros(1, opts.maxit);
end
outinfo.relerr = zeros(1, opts.maxit);

%%%  Main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ii = abs(X(:)) < sq2g_div_nG(:);

% Initializing conditions
conv = false;
it = 1;
while (it <= opts.maxit) && ~conv
    
    % Re-computing weights
    w = zeros(L2, 1);
    w(ii) = sq2g_times_nG(ii) - normsG2(ii) .* abs(X(ii));
    
    % Solving re-weighted L21 problem
    Xp = X;
    sigbp = sigb;
    [X, sigb, out_x] = wl1fista(Y, GlGl, replambdah, Gh0, SmSm, X, sigb, opts.b, w, Lip, opts.opts_fista);
    ii = abs(X(:)) < sq2g_div_nG(:);
    
    % Checking for convergence
    relerr = norm([X(:); sigb] - [Xp(:); sigbp]) / norm([Xp(:); sigbp]);
    if isnan(relerr)
        relerr = norm(X(:));
    end
    outinfo.relerr(it) = relerr; 
    conv = outinfo.relerr(it) < opts.tol;
    
    % Computing cost fn val
    if opts.computef
        [outinfo.fval(it), err, regval] = costfncelo(Y, X, sigb, opts.b, Gh0, replambdah, SmSm, gamma, sq2g_div_nG, normsG2, ii);
    end
    
    % Display iteration info
    if opts.verbose && mod(it,opts.dispfac) == 0
        disp(hline);
        disp(header);
        disp(subheader);
        disp(hline);
        strprog = sprintf(' %4d         %5.2e    %4d      ', it, relerr,length(out_x.relerr));
        if opts.computef
            strprog = [strprog  sprintf('%5.2e   %5.2e    %5.2e    %5d    (%5d)', ...
                outinfo.fval(it), err, regval, nnz(X), L2)];
        end
        disp(strprog);
        disp(hline);
    end    
    
    it = it + 1;
end

% The variance of the noise
Var_noise = sigb * opts.b(1,1);

% Preparing outputs
if opts.computef
    outinfo.fval = outinfo.fval(1:it-1);
end
outinfo.relerr = outinfo.relerr(1:it-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Auxiliary functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fval, err, reg] = costfncelo(Y, X0, sigb, b, Gh0, replambdah, SmSm, gamma, sq2g_div_nG, normsG2, ii)

N = size(Y,1)^(1/4);
L = size(Gh0, 1);

% Computing kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
v = compcov(X0, N, L, Gh0, SmSm, replambdah) + sigb*b;

err = sum((Y(:) - v(:)).^2);

reg = gamma;
reg(ii) = reg(ii) - 0.5 * normsG2(ii) .* (abs(X0(ii)) - sq2g_div_nG(ii)).^2;
reg = sum(reg);
if reg < 0
    reg = 0;
end

fval = 0.5 * err + reg;

function [X, sigb, out_x] = wl1fista(Y, GlGl, replambdah, Gh0, SmSm, X, sigb, b, w, Lip, opts_fista)

%%%  Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = size(Gh0,1);
N = sqrt(size(GlGl, 1));

% Step size and proximal operator threshold
step = 1/Lip;
wL = w./Lip;

% Prepare outputs
out_x.relerr = zeros(1, opts_fista.maxit);
if opts_fista.computef
    out_x.fval = zeros(1, opts_fista.maxit);
end

proxfn = @(x) proxglnn(x, wL);
costfn = @(x) costfngl(Y, x, sigb, b, Gh0, replambdah, w, SmSm);

if opts_fista.verbose
    hline = '---(inner it)----------------------------------------------------------------';
    if opts_fista.computef
        header = '   it          relerr        cost        err     ||X||_{W,1}   pixels   (grid)';
    else
        header = '   it          relerr          VarNoise          pixels   (grid)';
    end
    subheader = sprintf('(max=%4d)  (tol=%4.1e)', opts_fista.maxit, opts_fista.tol);
    plines = 0;
end

%%%  Main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializing conditions
t = 1;
XX = X;
sigbX = sigb;
conv = false;
it = 1;

% Stage 1: kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)' + s*B
v_keep = compcov(XX, N, L, Gh0, SmSm, replambdah) + sigbX*b;
% precompute a factor
factor = ((b(:)') * b(:));

while (it <= opts_fista.maxit) && ~conv
    
    % ------------------- rx minimization ---------------------------------
    
    % Computing gradient --------------------------------------------------
    
     % Stage 1: kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
     v = v_keep;
     
     % Stage 2: kr(kron(Gl, Gl),kron(Gl, Gl))' * (v(:) - ry)
     v = v(:) - Y(:);
     v = reshape(v, [N N N N]);
     v = munfold(v, [1 3], [2 4]);
     grad = GlGl' * v * GlGl;
    
    % Performing update --------------------------------------------------
    Xp = X;
    X = proxfn(XX - step * grad(:)); 
    
    % Determining whether to restart -------------------------------------
    if (XX(:) - X(:))'*(X(:) - Xp(:)) > 0
        disp('>>> restarted x <<<')
        t = 1;
        XX = X;
    else
    % Acceleration --------------------------------------------------
        tp = t;
        t = 0.5 * (1 + sqrt(1 + 4*tp^2));
        a = (tp - 1)/t;
        XX = X + a*(X - Xp);        
    end
    
    % ------------------- sigb minimization -------------------------------
    
    % keep the previous value ---------------------------------------------
    sigbp = sigb;
    
    % Performing update - explicit expression -----------------------------
    v_tmp = compcov(XX, N, L, Gh0, SmSm, replambdah);
    v = Y(:) - v_tmp(:);
    sigb = (b(:)')*v(:)/factor;
    sigb(sigb<0) = 0; % Non-negativity constraint 
    
    
    % --------- update sigb and keep for the next step --------------------
    
    v_keep = v_tmp + sigb*b;

    % Checking for convergence -------------------------------------------
    relerr = norm([X(:); sigb] - [Xp(:); sigbp]) / (norm([Xp(:); sigbp]) + eps);
    out_x.relerr(it) = relerr; 
    % Convergence conditions: small relative error & it > 1 
    conv = (out_x.relerr(it) < opts_fista.tol) & it > 1;
    
    % Computing cost fn val ----------------------------------------------
    if opts_fista.computef
        [out_x.fval(it), err, wxnrm] = costfn(X);
    end
    
    % Displaying progress ------------------------------------------------
    if opts_fista.verbose && mod(it,opts_fista.dispfac) == 0
        if mod(plines,20)==0
            disp(hline);
            disp(header);
            disp(subheader);
            disp(hline);
        end
        strprog = sprintf(' %4d         %5.2e         %5.2e    ', it, relerr, sigb * b(1,1));
        if opts_fista.computef
            strprog = [strprog  sprintf('%7.4e   %5.2e    %5.2e    %5d    (%5d)', ...
                out_x.fval(it), err, wxnrm, nnz(X(:)), L^2)];
        else
            strprog = [strprog  sprintf('  %5d    (%5d)', nnz(X(:)), L^2)];            
        end
        disp(strprog);
        plines = plines + 1;
    end
    
    it = it+1;
end

% Preparing outputs -----------------------------------------------------
if opts_fista.computef
    out_x.fval = out_x.fval(1:it-1);
end
out_x.relerr = out_x.relerr(1:it-1);

% --- LASSO ----------------------------------------------------

% Cost function
function [f, err, wxnrm] = costfngl(Y, X0, sigb, b, Gh0, replambdah, w, SmSm)

% (a) Computing GkGrx = kron(G, G) * Diag(rx)
N = numel(Y)^(1/4);
L = size(Gh0, 1);

% Computing kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
v = compcov(X0, N, L, Gh0, SmSm, replambdah)+ sigb*b;

err = sum((Y(:) - v(:)).^2);
wxnrm = abs(X0(:))'* w(:);
f = 0.5 * err + wxnrm;

% Proximal operator 
function X = proxglnn(X, w)
% NONNEGATIVE CASE:
X = X - w;
X(X(:) < 0) = 0;

% X(abs(X(:)) < w) = 0;
% X = sign(X) .* (abs(X) - w);