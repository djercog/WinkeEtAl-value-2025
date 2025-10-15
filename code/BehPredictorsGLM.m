% ============================================================
% Script for Ridge-glm on motor predictors at CS onset (Fig 1)
% 
% "Prefrontal neural geometry of associated cues guides learned motivated behaviors"
% Winke N, Luthi A, Herry C, Jercog D. 
% DOI: XXX
% github.com/djercog/WinkeEtAl-value-2025
% ============================================================

predictorNames = featureTable.Properties.VariableNames(1:end-2);
uniqueTypes = categories(featureTable.TrialType);
nTypes = numel(uniqueTypes);
uniqueSubjects = unique(featureTable.SubjectID);
nSub = numel(uniqueSubjects);
pairs = nchoosek(1:nTypes, 2);
nPairs = size(pairs, 1);
nPerm = 1000;               % num permutation test params
minTrialsPerSubject = 5;    % minimum number of trails
lambdaVal = 1;              %.5 %ridge regularization parameter (lambda)
pairResults_mouse = struct();
pairResults_mouse(nPairs).pairName = [];
% it over each pair of trial types (CSx/CSy)
for p = 1:nPairs
    pairTimer = tic;
    typeIdx1 = pairs(p,1);
    typeIdx2 = pairs(p,2);
    type1 = uniqueTypes{typeIdx1};
    type2 = uniqueTypes{typeIdx2};
    pairName = sprintf('%s_vs_%s', type1, type2);
    fprintf('Processing pair %d/%d: %s\n', p, nPairs, pairName);
    subjCoeffs = [];   
    subjCoefPerm = {};  
    for s = 1:nSub
        subIdx = (featureTable.SubjectID == uniqueSubjects(s)) & ...
                 (featureTable.TrialType == type1 | featureTable.TrialType == type2);
        if sum(subIdx) < minTrialsPerSubject
            warning('subject with not enough trials');
            continue;  
        end
        
        X_sub = featureTable{subIdx, predictorNames};
        X_sub = zscore(X_sub);% standardize predictors for subject!
        y_sub = double(featureTable.TrialType(subIdx) == type1);
        
        mdl = fitclinear(X_sub,y_sub,'Learner','logistic','Regularization','ridge','Lambda',lambdaVal,'Solver','lbfgs');
        coef_true_sub = mdl.Beta;  
        subjCoeffs = [subjCoeffs; coef_true_sub'];

        % Perm Subject 
        coef_perm_sub = zeros(length(predictorNames), nPerm);
        parfor perm = 1:nPerm
            y_perm = y_sub(randperm(length(y_sub)));
            try
                mdl_perm = fitclinear(X_sub, y_perm,'Learner','logistic','Regularization','ridge','Lambda',lambdaVal,'Solver', 'lbfgs');
                coef_perm_sub(:, perm) = mdl_perm.Beta;
            catch
                coef_perm_sub(:, perm) = NaN;  % if fails, record NaN (check after)
            end
        end
        subjCoefPerm{end+1} = coef_perm_sub;
    end  

    nSubjUsed = size(subjCoeffs,1);
    observedMedian = median(subjCoeffs, 1);
    % Null Distr for Group-Level Median
    nullMedian = nan(nPerm, length(predictorNames));
    for pred = 1:length(predictorNames)
        for permIdx = 1:nPerm
            tempVals = nan(nSubjUsed, 1);
            for subj = 1:nSubjUsed
                tempVals(subj) = subjCoefPerm{subj}(pred, permIdx);
            end
            nullMedian(permIdx, pred) = median(tempVals);
        end
    end
    % Group-Level p-values
    groupPvals = nan(1, length(predictorNames));
    for pred = 1:length(predictorNames)
        groupPvals(pred) = mean(abs(nullMedian(:, pred)) >= abs(observedMedian(pred)));
    end
    avgMedianCoeffs = observedMedian; 
    pairResults_mouse(p).pairName = pairName;
    pairResults_mouse(p).type1 = type1;
    pairResults_mouse(p).type2 = type2;
    pairResults_mouse(p).nSubjUsed = nSubjUsed;
    pairResults_mouse(p).avgMedianCoeffs = avgMedianCoeffs;
    pairResults_mouse(p).groupPvals = groupPvals;
end
coefMatrixMouse = nan(length(predictorNames), nPairs);
pvalMatrixMouse = nan(length(predictorNames), nPairs);
pairNamesCell = cell(nPairs,1);
for p = 1:nPairs
    pairNamesCell{p} = pairResults_mouse(p).pairName;
    coefMatrixMouse(:,p) = pairResults_mouse(p).avgMedianCoeffs(:);
    pvalMatrixMouse(:,p) = pairResults_mouse(p).groupPvals(:);
end