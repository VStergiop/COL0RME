function [tp, fp, fn, moldists] = molmatch(posU, posV, posUe, posVe, radius)

% Number of true molecules
R = length(posU);

% Number of estimated molecules
Re = length(posUe);

% First step: building lists of preferences -------------------------------

% Computing "preferences" of true molecules
preftrue = cell(1,R);
for r = 1:R
    
    % Computes all distances
    dist = sqrt((posU(r)-posUe).^2 + (posV(r)-posVe).^2);

    % Keeps only admissible distances, storing the molecule numbers
    cand = find(dist <= radius);
    dist = dist(dist <= radius);
    
    % Sorts distances in descending order and stores results
    [dist, is] = sort(dist, 'ascend');
    cand = cand(is);
    preftrue{r} = [cand; dist]; 
end

% Computing "preferences" of estimated molecules (as a sorted list)
prefest = cell(1,Re);
for re = 1:Re
    
    % Computes all distances
    dist = sqrt((posUe(re)-posU).^2 + (posVe(re)-posV).^2);

    % Keeps only admissible distances, storing the molecule numbers
    cand = find(dist <= radius);
    dist = dist(dist <= radius);
    
    % Sorts distances in descending order and stores results
    [dist, is] = sort(dist, 'ascend');
    cand = cand(is);
    prefest{re} = [cand; dist]; 
end

% Second step: matching candidates in possiblt several rounds -------------

% Stores the chosen matchings per estimate molecule
chosen = zeros(1,Re);
chosenpos = zeros(1,Re);
invchosen = zeros(1,R);

change = true;
while change
    
    change = false;
    
    % Loops over all true molecules
    for r = 1:R
        
        % If it still has choices, it "proposes" a match to the next
        % preferred estimate of its list
        if (invchosen(r) == 0) && ~isempty(preftrue{r})
            
            % Gets next prefered estimate
            pt = preftrue{r}(1,1);
            
            % If this estimate has not chosen any true molecule yet, it
            % considers matching with the current one
            if chosen(pt) == 0
                
                % Adds current true molecule as chosen one so far
                chosen(pt) = r;
                chosenpos(pt) = find(prefest{pt}(1,:) == r);
                invchosen(r) = pt;
                               
                % State has changed, must perform another iteration
                change = true;
                
            else  % Otherwise, this estimated molecule already considers another matching
                
                % So this will become a new matching only if it is better
                % for the estimated molecule
                pos = find(prefest{pt}(1,:) == r);
                if pos < chosenpos(pt)
                    
                    % In this case, the previously chosen true molecule has
                    % been rejected, and removes this estimated molecule
                    % from its list
                    invchosen(chosen(pt)) = 0;
                    preftrue{chosen(pt)} = preftrue{chosen(pt)}(:,2:end);
                    
                    % Updates the matching
                    chosen(pt) = r;
                    chosenpos(pt) = pos;
                    invchosen(r) = pt;
                    
                    % State has changed
                    change = true;
                elseif pos > chosenpos(pt)
                    % Otherwise, the true molecule has been rejected, and
                    % thus updates its list of preferences
                    preftrue{r} = preftrue{r}(:,2:end);
                    change = true;
                end
            end
        end
    end 
end

% Final step: computing results -------------------------------------------
moldists = zeros(1,R);
fn = zeros(1,R);
ifn = 0;
tp = zeros(2,R);
itp = 0;

% For each true molecule:
for r = 1:R
   % If it was not matched to any estimate molecule, then it is a false
   % negative    
   if invchosen(r) == 0
       moldists(r) = inf;
       ifn = ifn + 1;
       fn(ifn) = r;
   else
       % Otherwise, it is a true positive. In this case we store also the
       % corresponding estimated molecule
       moldists(r) = preftrue{r}(2,1);
       itp = itp + 1;
       tp(1, itp) = r;
       tp(2, itp) = invchosen(r);
   end
end
tp = tp(:, 1:itp);
fn = fn(1:ifn);

% False positives : were not retained by any true molecule
fp = 1:Re;
isfp = ~ismember(fp, tp(2,:));
fp = fp(isfp);
