function [x, it, time, crit, rv_sigb] = FBPD_VarN_2(x, f, g, h, w, opt)

% default inputs
if nargin < 2 || isempty(f),   f.fun = @(x) 0; f.prox = @(x,gamma) x;                                                   end
if nargin < 3 || isempty(g),   g.fun = @(x) 0; g.grad = @(x)       0; g.beta   = 0;                                     end
if nargin < 4 || isempty(h),   h.fun = @(y) 0; h.prox = @(y,gamma) y; h.dir_op = @(x) x; h.adj_op = @(y) y; h.beta = 1; end
if nargin < 5 || isempty(w),   1;  end
if nargin < 6 || isempty(opt), opt.tol = 2e-5; opt.iter = 100; opt.iter_max = 100;                                       end

% select the step-sizes
tau = 2 / (g.beta+2);
sigma = (1/tau - g.beta/2) / h.beta;

% initialize the dual solution
y = h.dir_op(x);

% initialize the variance of the noise
sigb = 0;

% execute the algorithm
time = zeros(1, opt.iter);
crit = gpuArray(zeros(1, opt.iter));
hdl = waitbar(0, 'Running FBPD...');
for it_max = 1:opt.iter_max
    
    for it = 1:opt.iter
        tic;
        % imagesc(x);
        % primal forward-backward step
        x_old = x;
        x = x - tau * ( g.grad(x,sigb) + h.adj_op(y) );
        x = f.prox(x, tau);

        % dual forward-backward step
        y = y + sigma * h.dir_op(2*x - x_old);
        y = y - sigma * h.prox(y/sigma, 1/sigma);

        % time and criterion
        time(it) = toc;
        crit(it) = f.fun(x) + g.fun(x,sigb) + h.fun(h.dir_op(x));

        % stopping rule
        if norm( x(:) - x_old(:) ) < opt.tol * norm( x_old(:) ) && it > 10
            break;
        end

        waitbar(it/opt.iter, hdl);
        rv_sigb = w.rv_sigb(sigb);
        strprog = sprintf('%2d     %2.4e     %2.4e    %2.4e', it, norm( x(:) - x_old(:) )/norm( x_old(:) ) ,crit(it), rv_sigb);
        disp(strprog);
    end
    
    %     sigb = sigb - w.tau * w.grad(x,sigb);
    %     sigb(sigb<0) = 0;
    sigbp = sigb;
    sigb = w.sigb(x);
    rv_sigb = w.rv_sigb(sigb);
    
    % stopping rule
    if norm( [x(:);sigb] - [x_old(:);sigbp] ) < opt.tol * norm( [x_old(:);sigbp] ) && it > 2
        break;
    end
end
close(hdl);
crit = crit(1:it);
time = cumsum(time(1:it));