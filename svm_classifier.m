%% load dataset
load('train_data_labels_ILDS.mat')

% numero de observaciones y caracteristicas
[N, d] = size(Xtrain);
% nombres de caracteristicas
nom_caract = {'Age','Female','TB','DB','Alkphos','Sgpt','Sgot',...
              'TP','ALB','A/R'};

% train - val partition
idx = randperm(size(Xtrain, 1));
Xtest = Xtrain(idx(400:463),:);
Lab_Xtest = Lab_Xtrain(idx(400:463));
Xtrain = Xtrain(idx(1:400),:);
Lab_Xtrain = Lab_Xtrain(idx(1:400));

% standarize
[Xtrain_scl,m,std] = standardize(Xtrain);
Xtest_scl = (Xtest - m) ./ std;

%% train linear svm
svm = fitcsvm(Xtrain_scl,Lab_Xtrain,'KFold',10,'Cost',[0 1;2 0]);
% test
acc_val_kc = zeros(svm.KFold,1);
f1_val_kc = zeros(svm.KFold,1);
for i = 1:svm.KFold
    pred = predict(svm.Trained{i}, Xtest_scl);
    [acc_val_kc(i),f1_val_kc(i)] = resumen(Lab_Xtest, pred);
end

%% train svm with gaussian kernel
H = 500;
hyper = [8+2*(rand(H,1)-0.5),7+3*(rand(H,1)-0.5)];
acc_val = zeros(H,1);
f1_val = zeros(H,1);
svms = {};
for j = 1:H
    svm = fitcsvm(Xtrain_scl,Lab_Xtrain,'KFold',10,...
                  'Weights',ones(size(Lab_Xtrain)) + 0.7*Lab_Xtrain,...
                  'KernelFunction','rbf',...
                  'KernelScale',hyper(j,1),'BoxConstraint',hyper(j,2));
    % test
    acc_val_kc = zeros(svm.KFold,1);
    f1_val_kc = zeros(svm.KFold,1);
    for i = 1:svm.KFold
        pred = predict(svm.Trained{i}, Xtest_scl);
        [acc_val_kc(i),f1_val_kc(i)] = resumen(Lab_Xtest, pred);
    end
    acc_val(j) = mean(acc_val_kc); f1_val(j) = mean(f1_val_kc);
    svms{j} = svm;
end

%% train svm with polinomial kernel
H = 10;
acc_val = zeros(H,1);
f1_val = zeros(H,1);
svms = {};
for j = 1:H
    svm = fitcsvm(Xtrain_scl,Lab_Xtrain,'KFold',10,...
                  'KernelFunction','polynomial',...
                  'PolynomialOrder',j);
    % test
    acc_val_kc = zeros(svm.KFold,1);
    f1_val_kc = zeros(svm.KFold,1);
    for i = 1:svm.KFold
        pred = predict(svm.Trained{i}, Xtest_scl);
        [acc_val_kc(i),f1_val_kc(i)] = resumen(Lab_Xtest, pred);
    end
    acc_val(j) = mean(acc_val_kc); f1_val(j) = mean(f1_val_kc);
    svms{j} = svm;
end
