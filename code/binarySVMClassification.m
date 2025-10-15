% ============================================================
% Binary SVM classification (used in Fig 2 & 3).
% 
% input:
%   dat: imDat struct with [nr trials x nr neurons x time bin]
%   classNameX: CS types to include in a class {'CSn'} (should work with multiple classes)
%   cellIndexs: vector of cell indices to use (-1 use all)
%   cvfold: number of cross-validation folds
%   nrRep: number of repetitions with random resampling trials
%   nrShuff: number of shuffle repetitions
% output:
%   accuracyMat: classification accuracies per time bin on diff repetitions 
%   accuracyMatSh: same but label shuffle
%   singleCellWeights: optional feature weights
%   svmModel: struct array of trained models for each time bin and repetition (NOTE: takes lots of ram, but needed for testing model generalization)
%
% "Prefrontal neural geometry of associated cues guides learned motivated behaviors"
% Winke N, Luthi A, Herry C, Jercog D. 
% DOI: XXX
% github.com/djercog/WinkeEtAl-value-2025
% ============================================================

function [accuracyMat,accuracyMatSh,singleCellWeights,svmModel]=binarySVMClassification(dat,className1,className2,cellIndexs,cvfold,nrRep,nrShuff)

datClass1=[];
for i=1:length(className1)
    datClass1=cat(1,datClass1,eval(['dat.',className1{i}]));
end
datClass2=[];
for i=1:length(className2)
    datClass2=cat(1,datClass2,eval(['dat.',className2{i}]));
end

nrTrial=min([size(datClass1,1),size(datClass2,1)]);
if cellIndexs==-1
    cellIndx=1:size(datClass1,2); %take all
else
    cellIndx=cellIndexs; 
    %Do not change model dimension: just set ignored variables to zero.
    datClass1(:,setdiff(1:size(datClass1,2),cellIndx),:)=0;
    datClass2(:,setdiff(1:size(datClass2,2),cellIndx),:)=0;
end
c=[1*ones(nrTrial,1);2*ones(nrTrial,1)];
datClass=[];
for j=1:length(dat.tm)
    for r=1:nrRep
        datAux=[];
        for l=1:size(datClass1,2)
            datAux(:,l)=[datClass1(randsample(size(datClass1,1),nrTrial),l,j);datClass2(randsample(size(datClass2,1),nrTrial),l,j)];
        end
        datClass{j}{r}=datAux;
    end
end
clear accuracyAux accWeights;
parfor j=1:length(dat.tm)
    auxV2=[];auxVW=[];svmModel2=[];
    for r=1:nrRep
        cp = cvpartition(c,'KFold',cvfold);
        datAux=datClass{j}{r};
        err = zeros(cp.NumTestSets,1);
        svmW=zeros(size(datAux,2),cp.NumTestSets);
        auxSvm=[];svmModel1=[];
        for g=1:cp.NumTestSets
            trIdx = cp.training(g);
            teIdx = cp.test(g);
            auxSvm = fitcsvm(datAux(trIdx,:),c(trIdx),'Solver','SMO');
            err(g)=sum(auxSvm.predict(datAux(teIdx,:))~=c(teIdx));
            svmW(:,g)=auxSvm.Beta;
            svmModel1(g).val=auxSvm;
        end
        auxVW(r).val=svmW;
        auxV2(r).val=100*(1-(sum(err)/sum(cp.TestSize)));
        svmModel2(r).val=svmModel1;
    end
    accuracyAux(j).val=[auxV2(:).val];
    accWeights(j).val=[auxVW(:).val]';
    svmModel3(j).val=svmModel2;
end
accuracyMat=vertcat(accuracyAux(:).val)';
singleCellWeights=accWeights;
%svmodels
for j=1:length(dat.tm)
    for r=1:nrRep
        for g=1:length(svmModel3(j).val(r).val)
            svmModel(j).val(r,g).mod=svmModel3(j).val(r).val(g).val;
        end
    end
end
accuracyMatSh=[];
if nrShuff>0
    clear accuracyAux;
    parfor j=1:length(dat.tm)
        auxV2=[];
        for r=1:nrShuff
            cp = cvpartition(c,'KFold',cvfold);
            datAux=[datClass1(randsample(size(datClass1,1),nrTrial),:,j);datClass2(randsample(size(datClass2,1),nrTrial),:,j)];
            datAux=datAux(randperm(size(datAux,1)),:);
            err = zeros(cp.NumTestSets,1);
            strTmp=[];
            for g=1:cp.NumTestSets
                trIdx = cp.training(g);
                teIdx = cp.test(g);
                auxSvm = fitcsvm(datAux(trIdx,:),c(trIdx),'Solver','SMO');
                err(g)=sum(auxSvm.predict(datAux(teIdx,:))~=c(teIdx));
            end
            auxV2(r).val=100*(1-(sum(err)/sum(cp.TestSize)));
        end
        accuracyAux(j).val=[auxV2(:).val];
    end
    accuracyMatSh=vertcat(accuracyAux(:).val)';
end

