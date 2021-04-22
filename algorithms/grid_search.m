function [f_mu_range, psnr_range] = grid_search(opts, sol, range, ynew, PSFfft, D, Mech, is, optsGen)

    N = size(Mech,1);
    L = size(Mech,2);
    
    PSFconj = conj(PSFfft);
    MechT = Mech';

    
    % initializations
    k=1;
    f_mu_range = zeros(size(range));
    psnr_range = zeros(size(range));

    for m = range
        
        if optsGen.BackEst==2
            lambda = opts.lambda;                       % the regularization parameter for b
            Lipb = opts.LipI + lambda*opts.LipDb;   % Lipschitz constant for b
            s = 1/Lipb;                         % step for b
        end
        
        % HERE IS THE INITIALIZATION OF x, x' and b
        x = zeros(L);
        if optsGen.BackEst==0
            b=0;
        elseif optsGen.BackEst==1
            b = median(ynew(:));
        elseif optsGen.BackEst==2
            b = ones(N)*median(ynew(:)); 
        end    
        
        Lip = opts.LipA + m*opts.LipD; % Lipschitz constant
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
                grad = real(ifft2(PSFconj.*fft2(MechT*(MHx - (ynew-b))*Mech))) + m*real(ifft2((D.DhT_DFT .* D.Dh_DFT + D.DvT_DFT .* D.Dv_DFT).*xfft));
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
                b(b<0) = 0;

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
                    b(b<0)= b(b<0)./(1+opts.alpha*s); 

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

        xfft = fft2(x);
        data_term = Mech * real(ifft2(PSFfft .* xfft)) * MechT - (ynew-b);
        f_mu_range(k)  = (1/2)*norm(data_term(:))^2 - (1/2)*opts.v_DP^2/optsGen.K*(N^2)*opts.est_var; % f(\mu)
        if optsGen.sim
            x_GT_cut = optsGen.x_GT(opts.pixels_fg+1:end-opts.pixels_fg,opts.pixels_fg+1:end-opts.pixels_fg);
            x_cut = x(opts.pixels_fg+1:end-opts.pixels_fg,opts.pixels_fg+1:end-opts.pixels_fg);
            mse = sum(sum((x_GT_cut - x_cut).^2))/(L^2);
            psnr_range(k) = 10 * log10( max(x_GT_cut(:))^2 / mse);
        end
        k = k+1;
    end    
    
    % Plots
    figure
    subplot 211
    plot(range,f_mu_range,'Linewidth',3)
    title('Grid search VS Discrepancy principle')
    ylabel('f(\mu)')
    xlabel('\mu')
    hold on;
    plot(range,zeros(size(range)),'--')
    hold on;
    plot(sol.mu,sol.f_mu, 'rx', 'Linewidth', 3)
    if optsGen.sim
        subplot 212
        plot(range,psnr_range,'Linewidth',3)
        title('Grid search VS Discrepancy principle')
        ylabel('PSNR')
        xlabel('\mu')
        hold on;
        plot(sol.mu,sol.psnr, 'rx', 'Linewidth', 3)
    end
end