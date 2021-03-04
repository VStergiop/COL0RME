classdef fdgradient    
    %fdgradient defines an finite difference operator for the computational
    %approximation of an n-dimensional gradient. The gradient has to be
    %initialized with signal dimensions "dim" and the corresponding
    %stepsizes "hvec"
    %
    %@author: Martin Benning
    
    properties(SetAccess=protected, GetAccess=protected)            
        dim %Dimension of input data
        mult %Nr. of partial derivatives
        transp %Indicator of matrix transposition
    end
    
    properties(SetAccess=protected, GetAccess=public)
        G %Gradient matrix
        hvec %Stepsize vector
    end
    
    methods
        function obj = fdgradient(sizevec, stepsize)
            if nargin == 0
                obj.dim = [64 64];
                obj.hvec = ones(2, 1);
            elseif nargin == 1
                obj.dim = sizevec;
                if (numel(sizevec) == 2) && (sizevec(2) == 1)
                    dimnr = 1;
                else
                    dimnr = numel(sizevec);
                end
                obj.hvec = ones(dimnr, 1);
            else
                obj.dim = sizevec;
                obj.hvec = stepsize;
            end
            obj.transp = 0; %matrix is not transposed                      
            %Set up N-D finite (forward) difference matrix via subsequent
            %application of the Kronecker product                        
            obj.G = buildGradient(obj);            
            obj.mult = max(size(obj.G))/min(size(obj.G));
        end    
        function P = mtimes(obj, B)
            %Redefinition of the matrix multiplication with respect to the
            %dimension of the finite differences gradient
            datadim = size(B);   
            if obj.transp == 0
                if ~isequal(datadim(1:obj.mult), obj.dim(1:obj.mult)) ...
                        || (numel(datadim) > (numel(obj.dim) + 1))
                    error('Matrix dimensions must agree!')
                else
                    if (numel(datadim) == (numel(obj.dim(1:obj.mult)) + 1))
                        frames = datadim(numel(datadim));
                    else
                        frames = 1;
                    end              
                    P = squeeze(reshape(obj.G*reshape(B, [prod(obj.dim) ...
                        frames]), [obj.dim obj.mult frames]));                                            
                end
            else
                if ~isequal(datadim(1:obj.mult), obj.dim(1:obj.mult)) ...
                        || (numel(datadim) > (numel(obj.dim) + 2))
                    error('Matrix dimensions must agree!')
                else
                    if (numel(datadim) == (numel(obj.dim(1:obj.mult)) ...
                            + 2)) || ((obj.mult == 1) && (datadim(2) ~= 1))
                        frames = datadim(numel(datadim));
                    else
                        frames = 1;
                    end
                    P = squeeze(reshape(obj.G'*reshape(B, [ ...
                        prod(obj.dim)*obj.mult frames]), [obj.dim ...
                        frames]));
                end
            end
        end
        function obj = transpose(obj)
            obj.transp = 1;
        end
        function obj = ctranspose(obj)
            obj.transp = 1;
        end
    end
    
    methods(Access = protected)
        function G = buildGradient(obj)
            %Set up N-D finite (forward) difference matrix via subsequent
            %application of the Kronecker product
            G = 1;
            if obj.dim(2) == 1
                noe = 1;
            else
                noe = numel(obj.dim);
            end
            for i=1:noe
                e = ones(obj.dim(i), 1);
                D = 1/obj.hvec(i) * spdiags([-e e], 0:1, obj.dim(i), ...
                    obj.dim(i));
                D(obj.dim(i), :) = 0;
                rowelem = size(G, 1)/(i - 1);
                if i == 1
                    G = kron(G, D);
                else
                    T = [];
                    for j=1:(i - 1)
                        T = [T; kron(speye(obj.dim(i)), ...
                            G(((j - 1)*rowelem + 1):(j*rowelem), :))];
                    end
                    G = [T; kron(D, speye(prod(obj.dim(1:(i - 1)))))];
                end
            end
        end
    end
    
end

