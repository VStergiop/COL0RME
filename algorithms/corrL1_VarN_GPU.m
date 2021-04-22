function [X, Var_noise] = corrL1_VarN_GPU(Y, Gl, Lip, q, gamma, opts)

% Display options
defopts.verbose = true;     % Verbose mode
defopts.dispfac = 1;        % Displays info only at multiples of this number
defopts.computef = true;

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
if nargin < 6 || isempty(opts)
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

Gl = single(Gl);
GlkGl = gpuArray(kron(Gl,Gl));
GlGl=gpuArray(khatrirao(Gl,Gl));

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
    X = zeros(L2, 1);
end
    [X, sigb, ~] = wl1fista(Y, GlkGl, GlGl, X, sigb, opts.b, gamma, Lip, opts.opts_fista); 
    if opts.verbose    
        outinfo.fval = costfn(Y, X, sigb, opts.b, GlkGl, gamma);
        fprintf('Computed initial point, cost fn value = %5.2e \n',outinfo.fval);
    end

% The variance of the noise
Var_noise = sigb * opts.b(1,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Auxiliary functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fval, err, reg] = costfn(Y, X0, sigb, b, GlkGl, gamma)

Y = gpuArray(Y);

% Computing kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
% v = compcov(X0, N, L, Gh0, SmSm, replambdah);
v = GlkGl.*X0';
v = v*GlkGl' + sigb*b;

err = sum((Y(:) - v(:)).^2);
reg = gamma*sum(abs(X0(:)));

fval = 0.5 * err + reg;

function [X, sigb, out_x] = wl1fista(Y, GlkGl, GlGl, X, sigb, b, w, Lip, opts_fista)

%%%  Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L = sqrt(size(X,1));
N = sqrt(size(GlGl, 1));

% Step size and proximal operator threshold
step = 1/Lip;
wL = w/Lip;

proxfn = @(x) proxglnn(x, wL);

if opts_fista.verbose
    hline = '---(inner it)----------------------------------------------------------------';
    if opts_fista.computef
        header = '   it          relerr        cost        err     ||X||_{1}   pixels   (grid)';
    else
        header = '   it          relerr          VarNoise      pixels   (grid)';
    end
    subheader = sprintf('(max=%4d)  (tol=%4.1e)', opts_fista.maxit, opts_fista.tol);
    plines = 0;
end

%%%  Main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializing conditions
t = 1;
X = single(X);
X = gpuArray(X);
XX = X;
conv = false;
it = 1;

% Stage 1: kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)' + s*B
v_keep = GlkGl.*XX';
v_keep = v_keep*GlkGl'+ sigb*b;
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
    v_tmp = GlkGl.*XX';
    v_tmp = v_tmp*GlkGl';
    v = Y(:) - v_tmp(:);
    sigb = (b(:)')*v(:)/factor;
    sigb(sigb<0) = 0; 
    
    
    % --------- update sigb and keep for the next step --------------------
    
    v_keep = v_tmp + sigb*b;
   
    % Checking for convergence
    relerr_x = norm(X(:) - Xp(:)) / norm(Xp(:));
    relerr_b = abs(sigb - sigbp) / abs(sigbp);
    out_x.relerr(it) = relerr_x; 
    % Convergence conditions: small relative error & it > 1 
    conv = (out_x.relerr(it) < opts_fista.tol) & (relerr_b < opts_fista.tol) & it > 1;
    
    
    % Computing cost fn val ----------------------------------------------
    if opts_fista.computef
        [out_x.fval(it), err, reg] = costfn(Y, X, sigb, b, GlkGl, w);
    end
    
    % Displaying progress ------------------------------------------------
    if opts_fista.verbose && mod(it,opts_fista.dispfac) == 0
        if mod(plines,20)==0
            disp(hline);
            disp(header);
            disp(subheader);
            disp(hline);
        end
        strprog = sprintf(' %4d         %5.2e         %5.2e    ', it, relerr_x, sigb * b(1,1));
        if opts_fista.computef
            strprog = [strprog  sprintf('%7.4e   %5.2e    %5.2e    %5d    (%5d)', ...
                out_x.fval(it), err, reg, nnz(X(:)), L^2)];
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

% Proximal operator 
function X = proxglnn(X, w)
% NONNEGATIVE CASE:
X = X - w;
X(X(:) <0) = 0;
% X(abs(X(:)) < w) = 0;
% X = sign(X) .* (abs(X) - w);
