function [lmb, v] = pwritr(v, Gh0, GlGl, replambdah, SmSm, maxit, tol, verbose)

if nargin < 6
    maxit = 100;
end

if nargin < 7
    tol = 1e-4;
end

if nargin < 8
    verbose = 1;
end

L = size(Gh0, 1);
N = sqrt(size(GlGl, 1));
q = L/N;

if verbose
    disp('-------------------------------------------')
    disp('it         lambda          err     time (s)')
    disp('-------------------------------------------')
end

for it = 1 : maxit
    
    inttimer = tic;
    
    % Apply matrix (kron(Gl, Gl)'*kron(Gl, Gl)) .* (kron(Gl, Gl)'*kron(Gl, Gl))
    vp = v;    
    % (a) Computing GkGrx = kron(G, G) * Diag(rx)
    GkGrx = zeros(N^2, L^2);

    % Computing kron(Gl, Gl) * Diag(vp) * kron(Gl, Gl)'
    w = compcov(vp, N, L, Gh0, SmSm, replambdah) ; 
    
    % Computing: kr(kron(Gl, Gl),kron(Gl, Gl))' * w(:)
    w = reshape(w, [N N N N]);
    w = munfold(w, [1 3], [2 4]);
    v = GlGl' * w * GlGl;    
    
    % Computing top eigenvalue
    lmb = norm(v(:));
    v = v / lmb;
    
    % Check for convergence (vectors have unit norm)
    err = norm(v(:) - vp(:));
    if err < tol 
        break
    end

    inttimer = toc(inttimer);
    
    % Display iteration info
    if verbose
        fprintf('%4d      %4.2e      %4.2e     %3.1f \n', it, lmb, err, inttimer);
    end
end
