function psnr_col0rme = computePSNR(x_GT_cut,int_x)
    L =size(x_GT_cut,1);
    mse = sum(sum((x_GT_cut - int_x).^2))/(L^2);
    psnr_col0rme = 10 * log10( max(x_GT_cut(:))^2 / mse);
end