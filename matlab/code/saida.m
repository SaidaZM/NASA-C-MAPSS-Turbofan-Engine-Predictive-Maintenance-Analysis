%% ==========================================================
%        PROJET FMDS & MAINTENANCE PREDICTIVE
%        Dataset : NASA C-MAPSS (FD001)
%        Auteur : Saida Zmitri
% ==========================================================

clc; clear; close all;

%% === 1. Charger les données ===============================
data = readtable('..\data\train_FD001.txt', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
data(:, all(ismissing(data))) = []; % Supprimer colonnes vides

% Renommer les colonnes
colNames = ["id","cycle","setting1","setting2","setting3", ...
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10", ...
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"];
data.Properties.VariableNames = colNames;

disp('✅ Données chargées');

%% === 2. Nettoyage des valeurs manquantes =================
for i = 1:max(data.id)
    idx = data.id == i;
    for j = 4:26 % settings + capteurs
        col = data{idx,j};
        if any(isnan(col))
            data{idx,j} = fillmissing(col,'linear');
        end
    end
end

%% === 3. Détection et traitement des outliers ============
X_raw = data{:,4:26};
mu = mean(X_raw);
sigma = std(X_raw);
X_clean = X_raw;

for j = 1:size(X_raw,2)
    outlier_idx = abs(X_raw(:,j) - mu(j)) > 3*sigma(j);
    X_clean(outlier_idx,j) = mu(j); % remplacer par la moyenne
end

data{:,4:26} = X_clean;

%% === 4. Normalisation sécurisée ==========================
X_norm = X_clean;
for j = 1:size(X_clean,2)
    col = X_clean(:,j);
    minVal = min(col);
    maxVal = max(col);
    if maxVal == minVal
        X_norm(:,j) = 0.5; % colonne constante
    else
        X_norm(:,j) = (col - minVal) / (maxVal - minVal);
    end
end
data{:,4:26} = X_norm;

%% === 5. Calcul de la RUL ================================
lastCycle = varfun(@max, data, 'InputVariables','cycle','GroupingVariables','id');
data = innerjoin(data, lastCycle(:,{'id','max_cycle'}),'Keys','id');
data.RUL = data.max_cycle - data.cycle;

%% === 6. Préparation des features et target =============
X = data{:,4:26};
Y = data.RUL;

valid_idx = all(~isnan(X),2) & ~isnan(Y);
X = X(valid_idx,:);
Y = Y(valid_idx);

%% === 7. Analyse FMDS de base ============================
nEngines = numel(unique(data.id));
fprintf('\nNombre de moteurs : %d\n', nEngines);

MTBF = mean(lastCycle.max_cycle);
MTTR = 5; % cycles
Disponibilite = MTBF/(MTBF+MTTR);
fprintf('MTBF estimé : %.2f cycles\n', MTBF);
fprintf('Disponibilité estimée : %.2f %%\n', Disponibilite*100);

% Fonction de fiabilité Weibull
t = 0:1:MTBF;
beta = 2; eta = MTBF;
R = exp(-(t/eta).^beta);
figure;
plot(t,R,'LineWidth',2);
xlabel('Cycles'); ylabel('R(t)');
title('Fonction de Fiabilité - Loi de Weibull');
grid on;

%% === 8. Corrélation capteurs ============================
corrMatrix = corr(X);
figure;
imagesc(corrMatrix); colorbar;
title('Corrélation entre capteurs');
xlabel('Capteurs'); ylabel('Capteurs');

%% === 9. Modèle de régression linéaire ===================
fprintf('\nEntraînement du modèle de régression linéaire...\n');
mdl_lin = fitlm(X,Y);
Y_pred_lin = predict(mdl_lin,X);
RMSE_lin = sqrt(mean((Y - Y_pred_lin).^2));
fprintf('RMSE régression linéaire : %.2f cycles\n', RMSE_lin);

%% === 10. Random Forest (TreeBagger) =====================
fprintf('\nEntraînement du modèle Random Forest...\n');
nTrees = 100;
mdl_rf = TreeBagger(nTrees,X,Y,'Method','regression','OOBPrediction','On');
Y_pred_rf = predict(mdl_rf,X);
RMSE_rf = sqrt(mean((Y - Y_pred_rf).^2));
fprintf('RMSE Random Forest : %.2f cycles\n', RMSE_rf);

%% === 11. Visualisation RUL pour un moteur =================
engine_id = 1;
subset = data(data.id==engine_id,:);
subset_idx = all(~isnan(subset{:,4:26}),2);
subset = subset(subset_idx,:);
Y_true = subset.RUL;

Y_pred_lin_sub = predict(mdl_lin, subset{:,4:26});
Y_pred_rf_sub = predict(mdl_rf, subset{:,4:26});

figure;
plot(subset.cycle,Y_true,'b-','LineWidth',2); hold on;
plot(subset.cycle,Y_pred_lin_sub,'r--','LineWidth',2);
plot(subset.cycle,Y_pred_rf_sub,'g-.','LineWidth',2);
xlabel('Cycle'); ylabel('RUL');
legend('RUL réel','RUL Linéaire','RUL Random Forest');
title(['RUL réel vs prédits - Moteur ', num2str(engine_id)]);
grid on;

%% === 12. Classification état sain / défaillant =========
labels = double(Y<30); % 1 = near failure, 0 = healthy
mdl_svm = fitcsvm(X,labels);
Y_pred_svm = predict(mdl_svm,X);

figure;
confusionchart(labels,Y_pred_svm,'Title','Diagnostic : état sain vs défaillant');

%% === 13. Synthèse ========================================
fprintf('\n===== SYNTHÈSE =====\n');
fprintf('MTBF moyen               : %.2f cycles\n', MTBF);
fprintf('MTTR supposé             : %.2f cycles\n', MTTR);
fprintf('Disponibilité            : %.2f %%\n', Disponibilite*100);
fprintf('RMSE régression linéaire : %.2f cycles\n', RMSE_lin);
fprintf('RMSE Random Forest       : %.2f cycles\n', RMSE_rf);
disp('Interprétation :');
disp('RMSE faible → bon modèle de pronostic. Disponibilité >90% → bon comportement global.');
