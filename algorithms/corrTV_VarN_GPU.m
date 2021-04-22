function [x, it, time, crit, rv_sigb] = corrTV_VarN_GPU(Y, Gl, Lip, lambda)

    % constraint
    f.prox = @(x,tau) project_box(x, 0, inf);
    
    % data fidelity
    L = size(Gl,2);
    
    Gl = single(Gl);
    GlkGl = gpuArray(kron(Gl,Gl));
    GlGl = gpuArray(khatrirao(Gl,Gl));
    B = eye(numel(Y)^(1/2))/numel(Y)^(1/2);
    factor = ((B(:)') * B(:));
    g.grad = @(x,sigb) GradLSx(Y, GlkGl, GlGl, x, sigb, B);
    g.beta = Lip;

    % criteria
    f.fun = @(x) indicator_box(x, 0, inf);
    g.fun = @(x, sigb) CostLS(Y, x, GlkGl, sigb, B);
    
    % forward finite differences (with Neumann boundary conditions)
    hor_forw = @(x) [x(:,2:end,:)-x(:,1:end-1,:), zeros(size(x,1),1,size(x,3))]; % horizontal
    ver_forw = @(x) [x(2:end,:,:)-x(1:end-1,:,:); zeros(1,size(x,2),size(x,3))]; % vertical

    % backward finite differences (with Neumann boundary conditions)
    hor_back = @(x) [-x(:,1,:), x(:,1:end-2,:)-x(:,2:end-1,:), x(:,end-1,:)];    % horizontal
    ver_back = @(x) [-x(1,:,:); x(1:end-2,:,:)-x(2:end-1,:,:); x(end-1,:,:)];    % vertical

    % direct and adjoint operators
    h.dir_op = @(x) cat( 4, hor_forw(x), ver_forw(x) );
    h.adj_op = @(y) hor_back( y(:,:,:,1) ) + ver_back( y(:,:,:,2) );

    % operator norm
    h.beta = 8;
   
    % proximity operator
    h.prox = @(y,gamma) prox_L2(y, gamma*lambda, 4);

    % criterion
    h.fun = @(y) fun_L2(y, lambda, 4);
    
    %% function for the variance of the noise

    w.grad = @(x,sigb) GradLSs(Y, GlkGl, x, sigb, B);
    w.sigb = @(x) EEs(Y, GlkGl, x, B, factor);
    w.rv_sigb = @(sigb) sigb * B(1,1);

    %% minimization
    x0 = zeros(L,L);
    [x, it, time, crit, rv_sigb] = FBPD_VarN(x0, f, g, h, w);    
    


function grad = GradLSx(Y, GlkGl, GlGl, X, sigb, B)

    N = sqrt(size(GlGl, 1));
    
    X = gpuArray(single(X(:)));
    v = GlkGl.*X';
    v = v*GlkGl' + sigb*B;

    % Stage 2: kr(kron(Gl, Gl),kron(Gl, Gl))' * (v(:) - ry)
    v = v(:) - Y(:);
    v = reshape(v, [N N N N]);
    v = munfold(v, [1 3], [2 4]);
    grad = GlGl' * v * GlGl;
    
function grad = GradLSs(Y, GlkGl, X, sigb, B)
    
    X = gpuArray(single(X(:)));
    v = GlkGl.*X';
    v = v*GlkGl' + sigb*B;

    % Stage 2: kr(kron(Gl, Gl),kron(Gl, Gl))' * (v(:) - ry)
    v = v(:) - Y(:);
    grad = B(:)'*v;
    
function sigb = EEs(Y, GlkGl, X, B, factor)
    X = gpuArray(single(X(:)));
    temp = GlkGl.*X';
    temp = temp*GlkGl';
    temp = Y(:) - temp(:);
    sigb = (B(:)')*temp(:)/factor;
    sigb(sigb<0) = 0; 


function err = CostLS(Y, X0, GlkGl, sigb, B)

    Y = gpuArray(Y);

    % Computing kron(Gl, Gl) * Diag(rx) * kron(Gl, Gl)'
    v = GlkGl.*X0(:)';
    v = v*GlkGl' + sigb*B;

    err = 0.5*sum((Y(:) - v(:)).^2);
    
     
    