function plot_results_COL0RME(Y_cut,supp_x_cut,int_x,b,opts,x_GT_cut,JacInd,Psnr,DP,var_noise,GT_var_noise)            
    

    figure
    subplot 321
    imagesc(Y_cut)
    colormap(gca,'hot')
    pbaspect([1 1 1])
    colorbar   
    title('Mean of the stack')
    subplot 322
    imagesc(x_GT_cut)
    colormap(gca,'hot')
    pbaspect([1 1 1])
    colorbar   
    title('Ground Truth')
    subplot 323
    imagesc(supp_x_cut)
    colormap(gca,'hot')
    pbaspect([1 1 1])
    title(['Estimated support, JI:' num2str(JacInd,'%.2f')])
    colorbar
    subplot 324
    imagesc(int_x,[0 max(x_GT_cut(:))])
    colormap(gca,'hot')
    pbaspect([1 1 1])
    title(['Estimated intensity, PSNR:' num2str(Psnr,'%.2f') 'dB'])
    colorbar
    if opts.BackEst == 2
        subplot 325
        imagesc(b)
        colormap(gca,'hot')
        pbaspect([1 1 1])
        title('Estimated background')
        colorbar
    end

end


