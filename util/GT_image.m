function gt = GT_image(Fluo, Cam, Grid, srf, indsY, K)
    
    ix = [(indsY{1}(1)-1)*srf+1 indsY{1}(end)*srf];
    iy = [(indsY{2}(1)-1)*srf+1 indsY{2}(end)*srf];
    
    % Generate ground truth images using all the frames    
    [GT_intensities_all,~,~] = gengroundtruth(Fluo, Cam, Grid, srf, ix, iy);
    gt.int_all = mean(GT_intensities_all,3);
    gt.pos_all = cell2mat(myind2sub(size(gt.int_all),find(GT_intensities_all(:) > 0)));
    
    % Generate ground truth images using K frames    
    Fluo.emitter_brightness = Fluo.emitter_brightness(1:K,:);
    [GT_intensities_K,~,~] = gengroundtruth(Fluo, Cam, Grid, srf, ix, iy);
    gt.int_K = mean(GT_intensities_K,3);
    gt.pos_K = cell2mat(myind2sub(size(gt.int_K),find(GT_intensities_K(:) > 0)));    

end
