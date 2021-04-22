function [X] = corrL1_noVarN_GPU(Y, Gl, Lip, q, gamma, opts)

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

% Initial solution
defopts.X0 = [];
defopts.Xpos = [];

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

% Initial solution

X = zeros(L2, 1);
[X, sigb, ~] = wl1fista(Y, GlkGl, GlGl, X, gamma, Lip, opts.opts_fista); 
if opts.verbose    
    outinfo.fval = costfn(Y, X, GlkGl, gamma);
    fprintf('Computed initial point, cost fn value = %5.2e \n',outinfo.fval);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Auxiliary functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fval, err, reg] = costfn(Y, X0, GlkGl, gamma)

Y = gpuArray(Y);

% Computing kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
% v = compcov(X0, N, L, Gh0, SmSm, replambdah);
v = GlkGl.*X0';
v = v*GlkGl';

err = sum((Y(:) - v(:)).^2);
reg = gamma*sum(abs(X0(:)));

fval = 0.5 * err + reg;

function [X, sigb, out_x] = wl1fista(Y, GlkGl, GlGl, X, w, Lip, opts_fista)

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
        header = '   it          relerr        pixels   (grid)';
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

while (it <= opts_fista.maxit) && ~conv
    
    % ------------------- rx minimization ---------------------------------

    % Computing gradient --------------------------------------------------
    
    % Stage 1: kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
    v = GlkGl.*XX';
    v = v*GlkGl';

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

   
    % Checking for convergence
    relerr = norm(X(:) - Xp(:)) / norm(Xp(:));
    out_x.relerr(it) = relerr; 
    % Convergence conditions: small relative error & it > 1 
    conv = (out_x.relerr(it) < opts_fista.tol) & it > 1;
    
    
    % Computing cost fn val ----------------------------------------------
    if opts_fista.computef
        [out_x.fval(it), err, reg] = costfn(Y, X, GlkGl, w);
    end
    
    % Displaying progress ------------------------------------------------
    if opts_fista.verbose && mod(it,opts_fista.dispfac) == 0
        if mod(plines,20)==0
            disp(hline);
            disp(header);
            disp(subheader);
            disp(hline);
        end
        strprog = sprintf(' %4d         %5.2e      ', it, relerr);
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
