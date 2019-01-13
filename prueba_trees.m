clear
close all

load('train_data_labels_ILDS.mat')

% READ DATASET

% number of classes and features
[n1, N_feat] = size(Xtrain);
N_classes = 2;
% number of samples
N_data = n1;
% labels
Labels =1 + Lab_Xtrain;
X = Xtrain;

%Generacion de datasets train, test y val
[X_train, Labels_train, X_test, Labels_test, X_val, Labels_val]=Train_Test_Val(X,Labels);

% Tree classifier design
tree = fitctree(X_train,Labels_train);
view(tree,'mode','graph');
view(tree)

%---------------------------------------------
%------------------- PRUNE -------------------
%---------------------------------------------

alpha_vector = 0.0001:0.0001:0.1;

error_train_prune = zeros(1,length(alpha_vector));
error_val_prune = zeros(1,length(alpha_vector));
error_test_prune = zeros(1,length(alpha_vector));
tree = fitctree(X_train,Labels_train, 'prune', 'off');

i=1;

for alpha = 0.0001:0.0001:0.02
    
     % Tree classifier design
    
    tree3 = prune(tree,'alpha',alpha);
    
    % Measure Train error
    outputs = predict(tree3,X_train);
    Tree_Pe_train=sum(Labels_train ~= outputs)/length(Labels_train);
    error_train_prune(i) = Tree_Pe_train;
    % Measure Val error
    outputs = predict(tree3,X_val);
    Tree_Pe_val=sum(Labels_val ~= outputs)/length(Labels_val);
    error_val_prune(i) = Tree_Pe_val;
    % Measure Test error
    outputs = predict(tree3,X_test);
    Tree_Pe_test=sum(Labels_test ~= outputs)/length(Labels_test);
    error_test_prune(i) = Tree_Pe_test;
    
    if i==1
        error_min = Tree_Pe_val;
        error_min_alpha = alpha;
        error_min_tree = tree3;
    end

    if Tree_Pe_val < error_min
        
        error_min = Tree_Pe_val;
        error_min_alpha = alpha;
        error_min_tree = tree3;
    end
    
    i=i+1;
end

tree = error_min_tree;
view(tree,'mode','graph');
view(tree)

fprintf(1,'Alpha with min error = %g   \n', error_min_alpha)

% Measure Train error
outputs = predict(tree,X_train);
Tree_Pe_train=sum(Labels_train ~= outputs)/length(Labels_train);
fprintf('\n------- TREE CLASSIFIER ------------------\n')   
fprintf(1,' error Tree train = %g   \n', Tree_Pe_train)  
CM_Train=confusionmat(Labels_train,outputs)
% Measure Val error
outputs = predict(tree,X_val);
Tree_Pe_val=sum(Labels_val ~= outputs)/length(Labels_val);
fprintf('\n-------------------------\n')   
fprintf(1,' error Tree val = %g   \n', Tree_Pe_val)  
CM_Val=confusionmat(Labels_val,outputs)
% Measure Test error
outputs = predict(tree,X_test);
Tree_Pe_test=sum(Labels_test ~= outputs)/length(Labels_test);
fprintf('\n-------------------------\n')   
fprintf(1,' error Tree test = %g   \n', Tree_Pe_test)
CM_Test=confusionmat(Labels_test,outputs)

[F,P,R] = evaluation(CM_Test);

%% 

load('test_data_ILDS.mat')
outputs = predict(tree,Xtest);
 resultados = zeros(116,2);

 for i = 1:116
     
     resultados(i,1) = i;
     resultados(i,2) = outputs(i) - 1;
     
     
 end
 
 csvwrite('resultados_1.dat',resultados);
 %% 
 
 Md = fitensemble(Xtrain,Lab_Xtrain,'LogitBoost',100,'Tree');
 
 load('test_data_ILDS.mat')
outputs = predict(Md,Xtest);
 resultados = zeros(116,2);

 for i = 1:116
     
     resultados(i,1) = i;
     resultados(i,2) = outputs(i);
     
     
 end
 
 csvwrite('resultados_2.dat',resultados);

