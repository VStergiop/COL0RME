function [sol_DiscPrinc] = DiscrepancyPrinciple(opts, ynew, PSFfft, D, Mech, is, optsGen)

    N = size(Mech,1);
    L = size(Mech,2);
    
    PSFconj = conj(PSFfft);
    MechT = Mech';

    % Intializations
    sol_DiscPrinc.DP = 1;
    mu = opts.mu_init;                              % the initialization of the regularization parameter
    if optsGen.BackEst==2
        lambda = opts.lambda;                       % the regularization parameter for b
        Lipb = opts.LipI + lambda*opts.LipDb;       % Lipschitz constant for b
        s = 1/Lipb;                                 % step for b
    end
    if optsGen.compute
        mu_matr = zeros(opts.maxIt,1);
        f_mu_matr = zeros(opts.maxIt,1);
        b_matr = zeros(opts.maxIt,1);
    end
    
    % HERE IS THE INITIALIZATION OF x, x' and b
    x = zeros(L); 
    xd = zeros(L);
    f_mu=0;
    if optsGen.BackEst==0
        b=0;
    elseif optsGen.BackEst==1
        b = median(ynew(:));
    elseif optsGen.BackEst==2
        b = ones(N)*median(ynew(:)); 
    end
    
    withoutDP = false;
    conv_f = false;
    l = 1; 
    while  ((l<=opts.maxIt) && ~conv_f) 

        if mu<0 || mu>1000 || isnan(mu)
            withoutDP = true;
            
            %initialize everything
            mu = opts.mu_init;
            mu_p = opts.mu_init;
            f_mu=0;
            x = zeros(L); 
            xd = zeros(L);
            if optsGen.BackEst==0
                b=0;
            elseif optsGen.BackEst==1
                b = median(ynew(:));
            elseif optsGen.BackEst==2
                b = ones(N)*median(ynew(:)); 
            end 
        end
        
        Lip = opts.LipA + mu*opts.LipD; % Lipschitz constant
        t = 1/Lip;
        
        % estimate x and b
        conv_xb = false;
        it_xb = 1;
        while (it_xb<opts.maxInIt_xb) && ~conv_xb
            b_p_it_xb = b;
            x_p_it_xb = x;
            
            % estimate x
            conv = false;
            it = 1;
            while (it<opts.maxInIt) && ~conv
                x_p = x; % keep the previous solution 

                xfft = fft2(x);
                MHx = Mech * real(ifft2(PSFfft .* xfft)) * MechT;
                grad = real(ifft2(PSFconj.*fft2(MechT*(MHx - (ynew-b))*Mech))) + mu*real(ifft2((D.DhT_DFT .* D.Dh_DFT + D.DvT_DFT .* D.Dv_DFT).*xfft));
                x = x-t*grad; % the gradient descent update

                % implementation of the prox (2 cases x>=0 , x<0)
                x_tmp=x;
                x(x_tmp>=0)=x_tmp(x_tmp>=0)./(1+opts.alpha*t*is(x_tmp>=0));     
                x(x_tmp<0)=x_tmp(x_tmp<0)./(1+opts.alpha*t*(is(x_tmp<0)+1));    

                % check for convergence
                relerr = norm(x(:) - x_p(:)) / norm(x_p(:));
                if isnan(relerr)
                    relerr = norm(x(:));
                end
                conv = relerr < opts.tol;
                it = it+1;
            end
            
            % estimate b
            if optsGen.BackEst == 0 
                conv_xb = true;
            elseif optsGen.BackEst == 1
                xfft = fft2(x);
                MHx = Mech * real(ifft2(PSFfft .* xfft)) * MechT;
                b = (1/N^2)*sum(ynew(:) - MHx(:));
                b(b<0) = b/opts.alpha;

                % check for convergence
                relerr = (b - b_p_it_xb)/b_p_it_xb; % I am checking only b for conv. here
                if isnan(relerr)
                    relerr =b;
                end
                conv_xb = abs(relerr) < opts.tol;
            elseif optsGen.BackEst == 2
                conv = false;
                it = 1;
                xfft = fft2(x);
                MHx = Mech * real(ifft2(PSFfft .* xfft)) * MechT;
                while (it<opts.maxInIt) && ~conv
                    b_p = b; % keep the previous solution 
                    
                    bfft = fft2(b);
                    grad = (MHx+b-ynew) + lambda*real(ifft2((D.DhT_DFT_b .* D.Dh_DFT_b + D.DvT_DFT_b .* D.Dv_DFT_b).*bfft));
                    b = b-s*grad; % the gradient descent update

                    % implementation of the prox, only positivity constr.   
                    b(b<0)= b(b<0)./(1+s*opts.alpha); 

                    % check for convergence
                    relerr = norm(b(:) - b_p(:)) / norm(b_p(:));
                    if isnan(relerr)
                        relerr = norm(b(:));
                    end
                    conv = relerr < opts.tol;
                    it = it+1;
                end
                % check for convergence
                relerr = norm([x(:) ; b(:)] - [x_p_it_xb(:) ; b_p_it_xb(:)]) / norm([x_p_it_xb(:) ; b_p_it_xb(:)]);
                if isnan(relerr)
                    relerr =norm([x(:) ; b(:)]);
                end
                conv_xb = abs(relerr) < opts.tol;
            end
            it_xb = it_xb+1;
        end
        
        if withoutDP
            break;
        end
        
        % estimate x'
        ix = zeros(L);
        ix(x<0)=1;
        xfft = fft2(x);
        conv = false;
        it = 1;
        while (it<opts.maxInIt) && ~conv
            xd_p = xd; % keep the previous solution 

            xdfft = fft2(xd);
            MHxd = Mech * real(ifft2(PSFfft .* xdfft)) * MechT;
            grad_d = real(ifft2(PSFconj.*fft2(MechT*MHxd*Mech))) + mu*real(ifft2((D.DhT_DFT .* D.Dh_DFT + D.DvT_DFT .* D.Dv_DFT).*xdfft)) + real(ifft2((D.DhT_DFT .* D.Dh_DFT + D.DvT_DFT .* D.Dv_DFT).*xfft));
            xd = xd-t*grad_d; % the gradient descent update

            % implementation of the prox
            xd=xd./(1+opts.alpha*t*(is+ix));

            % check for convergence
            relerr = norm(xd(:) - xd_p(:)) / norm(xd_p(:));
            if isnan(relerr)
                relerr = norm(xd(:));
            end
            conv = relerr < opts.tol;
            it = it+1;
        end

        xfft = fft2(x);
        data_term = Mech * real(ifft2(PSFfft .* xfft)) * MechT - (ynew-b);
        f_mu = (1/2)*norm(data_term(:))^2-(1/2)*opts.v_DP^2/optsGen.K*(N^2)*opts.est_var; 
        f_mu_der = sum(sum(conj(xd).*real(ifft2(PSFconj.*fft2(MechT*data_term*Mech)))));  

        if optsGen.compute
            mu_matr(l)=mu; 
            f_mu_matr(l) = f_mu;
            b_matr(l) = mean(b(:));
        end
        mu_p = mu;
        mu = mu-f_mu/f_mu_der; % \mu update
        
        % check for convergence
        conv_f = abs(f_mu)< opts.tol_f;
        if optsGen.compute && mod(l,50)==0
            fprintf('DP, Iteration:%5d    Cost:%4.1e\n',l,f_mu);
        end
        l = l+1;
    end
  
    sol_DiscPrinc.l = l;
    if l == (opts.maxIt+1)
        warning('The algorithm did not converge')
    end
    
    if withoutDP
        warning('The Discrepancy Principle does not work')
        sol_DiscPrinc.DP =0; 
    end
    
    sol_DiscPrinc.x = x;
    sol_DiscPrinc.mu = mu;
    sol_DiscPrinc.f_mu = f_mu;
    sol_DiscPrinc.b = b;
    
    
    if optsGen.compute && ~withoutDP
        figure;
        subplot 221
        plot(mu_matr(1:l-1))
        title('\mu')
        subplot 222
        plot(f_mu_matr(1:l-1))
        title('f(\mu)')
        subplot 223
        plot(b_matr(1:l-1))
        title('b')
    end

end