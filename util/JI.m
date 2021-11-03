function [Jacc_col0rme_Diag,Jacc_col0rme_Ver] = JI(int_GT,supp_x, pixsiz, srf)

L = size(int_GT,1);
pos_GT = cell2mat(myind2sub([L L],find(int_GT > 0)));
pos_col0rme = cell2mat(myind2sub([L L],...
    find(supp_x > 0)));
radiustol = 40;  % 40: allows diagonal matching; 25: only vertical or horizontal matching
[tp, fp, fn, ~] = ...
    molmatch((pixsiz/srf)*(pos_GT(:,1)'-0.5), (pixsiz/srf)*(pos_GT(:,2)'-0.5), ...
    (pixsiz/srf)*(pos_col0rme(:,1)'-0.5), (pixsiz/srf)*(pos_col0rme(:,2)'-0.5), radiustol);
Jacc_col0rme_Diag.all = size(tp,2)/(size(tp,2) + length(fp) + length(fn));
Jacc_col0rme_Diag.tp = size(tp,2);
Jacc_col0rme_Diag.fp = length(fp);
Jacc_col0rme_Diag.fn = length(fn);

radiustol = 25;  % 40: allows diagonal matching; 25: only vertical or horizontal matching
[tp, fp, fn, ~] = ...
    molmatch((pixsiz/srf)*(pos_GT(:,1)'-0.5), (pixsiz/srf)*(pos_GT(:,2)'-0.5), ...
    (pixsiz/srf)*(pos_col0rme(:,1)'-0.5), (pixsiz/srf)*(pos_col0rme(:,2)'-0.5), radiustol);
Jacc_col0rme_Ver.all = size(tp,2)/(size(tp,2) + length(fp) + length(fn));
