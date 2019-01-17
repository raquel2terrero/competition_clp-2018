%% load dataset
load('test_data_ILDS.mat')
load('train_data_labels_ILDS.mat')

% numero de observaciones y caracteristicas
[N, d] = size(Xtrain);
% nombres de caracteristicas
nom_caract = {'Age', 'Female', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', ...
              'TP', 'ALB', 'A/R'};

% train - val partition
idx = randperm(size(Xtrain, 1));
Xval = Xtrain(idx(400:463),:);
Lab_Xval = Lab_Xtrain(idx(400:463));
Xtrain = Xtrain(idx(1:400),:);
Lab_Xtrain = Lab_Xtrain(idx(1:400));

%% train trees
CVO = cvpartition(Lab_Xtrain,'k',10);
acc = zeros(CVO.NumTestSets,1);
f1 = zeros(CVO.NumTestSets,1);
trees = {};
for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    tree = fitctree(Xtrain(trIdx,:), Lab_Xtrain(trIdx),...
                    'MaxNumSplits',5,...
                    'PruneCriterion','impurity',...
                    'PredictorNames',nom_caract);
    pred = predict(tree, Xtrain(teIdx,:));
    [a,f] = resumen(Lab_Xtrain(teIdx,:), pred);
    acc(i) = a; f1(i) = f;
    trees{i} = tree;
end

%% acc and f1-score on validation set
acc_val = zeros(CVO.NumTestSets,1);
f1_val = zeros(CVO.NumTestSets,1);
for i = 1:length(trees)
    pred = predict(trees{i}, Xval);
    [a,f] = resumen(Lab_Xval, pred);
    acc_val(i) = a; f1_val(i) = f;
end

maxk(f1_val,3)