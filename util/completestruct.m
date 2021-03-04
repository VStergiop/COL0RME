function s1 = completestructure(s1,s2)

fields = fieldnames(s2);

for f=1:length(fields)
    name = fields{f};
    if ~isfield(s1,name)
        s1.(name) = s2.(name);
    end
end    
