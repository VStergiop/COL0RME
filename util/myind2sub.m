function v = myind2sub(siz,ndx) 
[out{1:length(siz)}] = ind2sub(siz,ndx); 
if nargout > 1
    v = cell2mat(out);
else
    v = out;
end    