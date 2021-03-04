function [lmb, v] = pwritr_GPU(v, Gl, maxit, tol, verbose)

if nargin < 3
    maxit = 100;
end

if nargin < 4
    tol = 1e-4;
end

if nargin < 5
    verbose = 1;
end

N = size(Gl, 1);

Gl = single(Gl);
GlkGl = kron(Gl,Gl);
GlkGl = gpuArray(GlkGl);
GlGl = khatrirao(Gl,Gl);
GlGl=gpuArray(GlGl);

if verbose
    disp('-------------------------------------------')
    disp('it         lambda          err     time (s)')
    disp('-------------------------------------------')
end

v = single(v);
v = gpuArray(v); 

gradi = @(vp) grad(vp,GlkGl,GlGl,N);

for it = 1 : maxit
    
    inttimer = tic;
    vp = v(:);   
    
    % Apply matrix (kron(Gl, Gl)'*kron(Gl, Gl)) .* (kron(Gl, Gl)'*kron(Gl, Gl))
    v = gradi(vp);  
    
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


function v = grad(vp,GlkGl,GlGl,N)
    
    % Computing kron(Gl, Gl) * Diag(vp) * kron(Gl, Gl)'
    w = GlkGl.*vp';
    w = w*GlkGl';
    % Computing: kr(kron(Gl, Gl),kron(Gl, Gl))' * w(:)
    w = reshape(w, [N N N N]);
    w = munfold(w, [1 3], [2 4]);
    v = GlGl' * w * GlGl; 
