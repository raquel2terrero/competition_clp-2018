load('dataset/test_data_ILDS.mat')
load('dataset/train_data_labels_ILDS.mat')

% dividir por clases
Xtrain_san = Xtrain(Lab_Xtrain==0,:);
Xtrain_enf = Xtrain(Lab_Xtrain==1,:);
% numero de observaciones y caracteristicas
[N, d] = size(Xtrain);
% nombres de caracteristicas
nom_caract = {'Age', 'Female', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', ...
              'TP', 'ALB', 'A/R'};

%% HISTOGRAMAS GAUSSIANA
figure('name','Histograms')  
for i_feat=1:d
    subplot(2, d, i_feat)
    histfit(Xtrain_san(:,i_feat))
    grid
    zoom on
    title(['Sano: ' nom_caract{i_feat}])
end
for i_feat=1:d
    subplot(2, d, 10+i_feat)
    histfit(Xtrain_enf(:,i_feat))
    grid
    zoom on
    title(['Enfermo: ' nom_caract{i_feat}])
end
figure('name', 'Global Histogram')
for i_feat=1:d
    subplot(2,5,i_feat)
    histfit(Xtrain(:,i_feat))
    grid
    zoom on
    title(nom_caract{i_feat})
end

%% HISTOGRAMAS EXPONENCIAL
figure('name','Histogramas exponenciales')  
for i_feat=1:d
    subplot(2, d, i_feat)
    histfit(Xtrain_san(:,i_feat),[], 'exponential')
    grid
    zoom on
    title(['Sano: ' nom_caract{i_feat}])
end
for i_feat=1:d
    subplot(2, d, 10+i_feat)
    histfit(Xtrain_enf(:,i_feat), [], 'exponential')
    grid
    zoom on
    title(['Enfermo: ' nom_caract{i_feat}])
end
figure('name', 'Global Histogram Exponential')
for i_feat=1:d
    subplot(2,5,i_feat)
    histfit(Xtrain(:,i_feat), [], 'exponential')
    grid
    zoom on
    title(nom_caract{i_feat})
end

%% PLOTNORM
figure('name', 'Sanos')
for i_feat=1:d
    subplot(2,5,i_feat)
    qqplot(Xtrain_san(:,i_feat))
    grid
    title(nom_caract{i_feat})
end
figure('name', 'Enfermos')
for i_feat=1:d
    subplot(2,5,i_feat)
    qqplot(Xtrain_enf(:,i_feat))
    grid
    title(nom_caract{i_feat})
end
figure('name', 'Global')
for i_feat=1:d
    subplot(2,5,i_feat)
    qqplot(Xtrain(:,i_feat))
    grid
    title(nom_caract{i_feat})
end

%% Test de hipotesis
for i_feat=1:d
    i_feat
    [H_san,p_value] = chi2gof(Xtrain_san(:,i_feat),'Alpha',0.05)
    [H_enf,p_value] = chi2gof(Xtrain_enf(:,i_feat),'Alpha',0.05)
end

%% SCATTER PLOT
figure('name','Scatter Plot')
gplotmatrix(Xtrain,Xtrain,Lab_Xtrain,'bgr',[],[],'on','hist',...
            nom_caract,nom_caract);
zoom on

%% Empirical Comulative Distribution Function
for i_feat=1:d
    figure
    ecdf(Xtrain_san(:,i_feat))
    hold on
    ecdf(Xtrain_enf(:,i_feat))
    title([nom_caract{i_feat}])
    legend('sano','enfermo');
end
