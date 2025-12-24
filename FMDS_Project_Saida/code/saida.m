%% ==========================================================
%        PROJET FMDS OPTIMISÉ - MEILLEURS RÉSULTATS
%        Dataset : NASA C-MAPSS (FD001)
%        Optimisations : Feature selection, Hyperparameters, RUL capping
% ==========================================================
clear; clc; close all; rng(42); % Fixed seed pour reproductibilité

%% === 1. Charger les données ===============================
[file, path] = uigetfile({'*.txt','Text Files (*.txt)'}, 'Sélectionne train_FD001.txt');
if isequal(file,0)
    error('❌ Aucun fichier sélectionné.');
end
dataPath = fullfile(path,file);
disp(['📄 Fichier : ', dataPath]);

data = readtable(dataPath,'Delimiter',' ','MultipleDelimsAsOne',true);
data(:, all(ismissing(data))) = [];

colNames = ["id","cycle","setting1","setting2","setting3", ...
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10", ...
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"];

if width(data) >= 26
    data.Properties.VariableNames(1:26) = colNames;
else
    error('Dataset invalide.');
end
disp('✅ Données chargées.');

%% === 2. Prétraitement robuste ===============================
fprintf('🔧 Prétraitement...\n');

% Supprimer capteurs constants (variance nulle)
X_raw = data{:,4:26};
variances = var(X_raw, 'omitnan');
constant_cols = variances < 1e-6;
valid_sensor_cols = 4:26;
valid_sensor_cols(constant_cols) = [];
fprintf('   Capteurs constants supprimés : %d\n', sum(constant_cols));

% Interpolation valeurs manquantes par moteur
for i = 1:max(data.id)
    idx = data.id == i;
    for j = valid_sensor_cols
        col = data{idx,j};
        if any(isnan(col))
            data{idx,j} = fillmissing(col,'linear','EndValues','nearest');
        end
    end
end

% Outliers : Clipping à 4 sigmas au lieu de remplacement par moyenne
X_clean = data{:,valid_sensor_cols};
for j = 1:size(X_clean,2)
    col = X_clean(:,j);
    mu = mean(col,'omitnan');
    sigma = std(col,'omitnan');
    % Clipping au lieu de remplacement
    X_clean(:,j) = max(min(col, mu + 4*sigma), mu - 4*sigma);
end

% Normalisation Min-Max robuste
X_norm = zeros(size(X_clean));
for j = 1:size(X_clean,2)
    col = X_clean(:,j);
    p5 = prctile(col, 5);
    p95 = prctile(col, 95);
    if p95 > p5
        X_norm(:,j) = (col - p5) / (p95 - p5);
        X_norm(:,j) = max(0, min(1, X_norm(:,j))); % Clip [0,1]
    else
        X_norm(:,j) = 0.5;
    end
end
data{:,valid_sensor_cols} = X_norm;

%% === 3. Calcul RUL avec plafonnement (RUL capping) =========
[G, ids] = findgroups(data.id);
maxCyclePerID = splitapply(@max, data.cycle, G);
lastCycle = table(ids, maxCyclePerID, 'VariableNames', {'id','max_cycle'});
data = innerjoin(data, lastCycle, 'Keys','id');
data.RUL = data.max_cycle - data.cycle;

% RUL CAPPING : Limiter à 125 cycles (améliore la prédiction)
RUL_cap = 125;
data.RUL_capped = min(data.RUL, RUL_cap);
fprintf('   RUL plafonnée à %d cycles\n', RUL_cap);

%% === 4. Feature Engineering ================================
% Ajouter moyennes mobiles pour capturer tendances
window = 5;
data.s2_ma = movmean(data.s2, window);
data.s3_ma = movmean(data.s3, window);
data.s4_ma = movmean(data.s4, window);
data.s7_ma = movmean(data.s7, window);
data.s11_ma = movmean(data.s11, window);
data.s12_ma = movmean(data.s12, window);

% Ajouter écarts-types mobiles (volatilité)
data.s11_std = movstd(data.s11, window);
data.s12_std = movstd(data.s12, window);

%% === 5. Sélection des features importantes =================
% Utiliser toutes features + engineered
feature_cols = [valid_sensor_cols, width(data)-7:width(data)];
X = data{:,feature_cols};
Y = data.RUL_capped;

% Supprimer lignes avec NaN
valid_idx = all(~isnan(X),2) & ~isnan(Y);
X = X(valid_idx,:);
Y = Y(valid_idx);
fprintf('   Features utilisées : %d\n', size(X,2));
fprintf('   Observations valides : %d\n', size(X,1));

%% === 6. Split Train/Validation (80/20) =====================
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv),:);
Y_train = Y(training(cv));
X_val = X(test(cv),:);
Y_val = Y(test(cv));

%% === 7. Modèle Random Forest OPTIMISÉ =====================
fprintf('\n🌲 Entraînement Random Forest optimisé...\n');
mdl_rf_opt = TreeBagger(300, X_train, Y_train, ...
    'Method','regression', ...
    'OOBPrediction','On', ...
    'MinLeafSize',3, ...
    'NumPredictorsToSample','auto', ...
    'MaxNumSplits',100);

Y_pred_train = predict(mdl_rf_opt, X_train);
Y_pred_val = predict(mdl_rf_opt, X_val);

RMSE_train = sqrt(mean((Y_train - Y_pred_train).^2));
RMSE_val = sqrt(mean((Y_val - Y_pred_val).^2));
MAE_val = mean(abs(Y_val - Y_pred_val));

fprintf('   RMSE Train : %.2f cycles\n', RMSE_train);
fprintf('   RMSE Validation : %.2f cycles\n', RMSE_val);
fprintf('   MAE Validation : %.2f cycles\n', MAE_val);

%% === 8. Modèle Régression Linéaire (baseline) =============
fprintf('\n📈 Régression linéaire (baseline)...\n');
mdl_lin = fitlm(X_train, Y_train);
Y_pred_lin_val = predict(mdl_lin, X_val);
RMSE_lin_val = sqrt(mean((Y_val - Y_pred_lin_val).^2));
fprintf('   RMSE Linéaire Validation : %.2f cycles\n', RMSE_lin_val);

%% === 9. Classification optimisée (SVM avec RBF) ============
fprintf('\n🎯 Classification SVM optimisée...\n');
% Seuil plus conservateur : 50 cycles
threshold_failure = 50;
labels_train = double(Y_train < threshold_failure);
labels_val = double(Y_val < threshold_failure);

mdl_svm = fitcsvm(X_train, labels_train, ...
    'Standardize',true, ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...
    'BoxConstraint',1);

Y_pred_svm = predict(mdl_svm, X_val);

% Métriques de classification
TP = sum((labels_val == 1) & (Y_pred_svm == 1));
TN = sum((labels_val == 0) & (Y_pred_svm == 0));
FP = sum((labels_val == 0) & (Y_pred_svm == 1));
FN = sum((labels_val == 1) & (Y_pred_svm == 0));

Accuracy = (TP + TN) / (TP + TN + FP + FN);
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
F1 = 2 * Precision * Recall / (Precision + Recall);

fprintf('   Accuracy : %.2f%%\n', Accuracy*100);
fprintf('   Precision : %.2f%%\n', Precision*100);
fprintf('   Recall : %.2f%%\n', Recall*100);
fprintf('   F1-Score : %.2f%%\n', F1*100);

%% === 10. Analyse Weibull ROBUSTE ===========================
lifetimes = lastCycle.max_cycle;
lifetimes(lifetimes < 1) = 1; % Éviter valeurs nulles

try
    pd = fitdist(lifetimes,'Weibull');
    eta_hat = pd.A;
    beta_hat = pd.B;
catch
    [eta_hat, beta_hat] = wblfit(lifetimes);
end

MTBF = mean(lifetimes);
MTTR = 5;
Disponibilite = MTBF/(MTBF+MTTR);

fprintf('\n📊 Indicateurs FMDS:\n');
fprintf('   MTBF : %.2f cycles\n', MTBF);
fprintf('   Disponibilité : %.2f%%\n', Disponibilite*100);
fprintf('   Weibull η : %.2f\n', eta_hat);
fprintf('   Weibull β : %.2f\n', beta_hat);

%% === 11. Préparation visualisations ========================
% Moteur avec vie médiane pour visualisation
median_life = median(lifetimes);
[~, best_engine_idx] = min(abs(lifetimes - median_life));
engine_id = ids(best_engine_idx);

subset = data(data.id==engine_id,:);
subset_idx = all(~isnan(subset{:,feature_cols}),2);
subset = subset(subset_idx,:);

xgrid = linspace(0, max(lifetimes)*1.2, 1000);

% Tri 3-classes OPTIMISÉ
th1 = 80; th2 = 40; % Seuils ajustés
tri_true_val = zeros(numel(Y_val),1);
tri_true_val(Y_val <= th2) = 2;
tri_true_val(Y_val > th2 & Y_val <= th1) = 1;
tri_true_val(Y_val > th1) = 0;

tri_pred_val = zeros(numel(Y_pred_val),1);
tri_pred_val(Y_pred_val <= th2) = 2;
tri_pred_val(Y_pred_val > th2 & Y_pred_val <= th1) = 1;
tri_pred_val(Y_pred_val > th1) = 0;

%% === 12. DASHBOARD OPTIMISÉ =================================
fig = figure('Name','Dashboard FMDS Optimisé','Position',[50 50 1600 900]);

% 1. Fonction de fiabilité R(t) Weibull
subplot(2,3,1);
plot(xgrid, exp(-(xgrid/eta_hat).^beta_hat), 'b-', 'LineWidth', 2.5);
hold on;
plot(xgrid, exp(-xgrid/MTBF), 'r--', 'LineWidth', 1.5);
xlabel('Temps (cycles)'); ylabel('R(t)');
title('Fiabilité R(t) - Weibull vs Exponentielle');
legend('Weibull','Exponentielle','Location','best');
grid on; ylim([0 1]);

% 2. Matrice de confusion SVM
subplot(2,3,2);
cm = confusionchart(labels_val, Y_pred_svm);
cm.Title = sprintf('Confusion SVM (RUL<%.0f)\nAccuracy=%.1f%%', threshold_failure, Accuracy*100);

% 3. Distribution RUL (avant/après capping)
subplot(2,3,3);
histogram(data.RUL, 50, 'Normalization','pdf', 'FaceAlpha',0.5); hold on;
histogram(data.RUL_capped, 50, 'Normalization','pdf', 'FaceAlpha',0.5);
xlabel('RUL (cycles)'); ylabel('Densité');
title('Distribution RUL');
legend('RUL originale','RUL plafonnée','Location','best');
grid on;

% 4. Comparaison prédictions RUL (moteur représentatif)
subplot(2,3,[4,5]);
if ~isempty(subset)
    Y_true_sub = subset.RUL_capped;
    Y_pred_rf_sub = predict(mdl_rf_opt, subset{:,feature_cols});
    Y_pred_lin_sub = predict(mdl_lin, subset{:,feature_cols});
    
    plot(subset.cycle, Y_true_sub, 'b-', 'LineWidth', 3); hold on;
    plot(subset.cycle, Y_pred_rf_sub, 'g-.', 'LineWidth', 2);
    plot(subset.cycle, Y_pred_lin_sub, 'r--', 'LineWidth', 1.5);
    
    xlabel('Cycle'); ylabel('RUL (cycles)');
    legend('RUL réelle','Random Forest','Linéaire','Location','best');
    title(sprintf('Prédiction RUL - Moteur %d (vie médiane: %.0f cycles)', engine_id, median_life));
    grid on;
    
    text(0.7*max(subset.cycle), 0.85*max(Y_true_sub), ...
        sprintf('RMSE RF: %.2f\nRMSE Lin: %.2f\nAmélioration: %.1f%%', ...
        RMSE_val, RMSE_lin_val, 100*(RMSE_lin_val-RMSE_val)/RMSE_lin_val), ...
        'FontSize', 9, 'BackgroundColor', 'yellow', 'EdgeColor', 'black');
end

% 5. Feature Importance (top 10)
subplot(2,3,6);
importance = mdl_rf_opt.OOBPermutedPredictorDeltaError;
[sorted_imp, idx] = sort(importance, 'descend');
top_n = min(10, length(sorted_imp));
barh(sorted_imp(1:top_n));
set(gca, 'YTick', 1:top_n, 'YTickLabel', idx(1:top_n));
xlabel('Importance (OOB Error Δ)');
title('Top 10 Features');
grid on;

sgtitle('Dashboard FMDS Optimisé - Meilleurs Résultats', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% === 13. DASHBOARD 2 : Analyses Avancées ===================
fig2 = figure('Name','Analyses Avancées','Position',[100 100 1600 900]);

% 1. Résidus prédiction
subplot(2,3,1);
residuals = Y_val - Y_pred_val;
scatter(Y_pred_val, residuals, 10, 'filled', 'MarkerFaceAlpha',0.3);
hold on; yline(0, 'r--', 'LineWidth',2);
xlabel('RUL prédite'); ylabel('Résidus');
title('Analyse des résidus RF');
grid on;

% 2. Prédiction vs Réel (scatter)
subplot(2,3,2);
scatter(Y_val, Y_pred_val, 10, 'filled', 'MarkerFaceAlpha',0.3);
hold on; 
plot([0 max(Y_val)], [0 max(Y_val)], 'r--', 'LineWidth',2);
xlabel('RUL réelle'); ylabel('RUL prédite');
title('Prédiction vs Réalité');
axis equal; grid on;

% 3. Confusion 3-classes
subplot(2,3,3);
cm3 = confusionchart(categorical(tri_true_val, [0 1 2], {'Sain', 'Surveillance', 'Urgent'}), ...
                     categorical(tri_pred_val, [0 1 2], {'Sain', 'Surveillance', 'Urgent'}));
cm3.Title = 'Classification 3 niveaux';

% 4. Hazard rate Weibull
subplot(2,3,4);
h_weib = (beta_hat/eta_hat) * ((xgrid/eta_hat).^(beta_hat-1));
plot(xgrid, h_weib, 'b-', 'LineWidth', 2);
xlabel('Cycles'); ylabel('h(t)');
title('Taux de défaillance (Weibull)');
grid on;

% 5. Survival function (KM vs Weibull)
subplot(2,3,5);
[f_km, x_km] = ecdf(lifetimes, 'Function','survivor');
stairs(x_km, f_km, 'b-', 'LineWidth', 2); hold on;
plot(xgrid, 1 - wblcdf(xgrid, eta_hat, beta_hat), 'r--', 'LineWidth', 2);
xlabel('Cycles'); ylabel('S(t)');
title('Courbe de survie');
legend('Kaplan-Meier','Weibull','Location','best');
grid on;

% 6. Erreur absolue par niveau RUL
subplot(2,3,6);
bins = [0 40 80 125];
bin_labels = {'0-40','40-80','80-125'};
[~,~,bin_idx] = histcounts(Y_val, bins);
boxplot(abs(residuals), bin_idx, 'Labels', bin_labels);
ylabel('Erreur absolue (cycles)');
xlabel('Plage RUL réelle');
title('Distribution erreur par plage RUL');
grid on;

sgtitle('Analyses Avancées - Validation & Diagnostics', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% === 14. SYNTHÈSE FINALE ====================================
fprintf('\n========== SYNTHÈSE OPTIMISÉE ==========\n');
fprintf('Dataset : %d moteurs, %d observations\n', numel(ids), size(data,1));
fprintf('\n--- FMDS ---\n');
fprintf('MTBF        : %.2f cycles\n', MTBF);
fprintf('MTTR        : %.2f cycles\n', MTTR);
fprintf('Disponibilité : %.2f%%\n', Disponibilite*100);
fprintf('\n--- Weibull ---\n');
fprintf('η (scale)   : %.2f\n', eta_hat);
fprintf('β (shape)   : %.2f\n', beta_hat);
fprintf('\n--- Performance Prédictive ---\n');
fprintf('RMSE RF Val      : %.2f cycles\n', RMSE_val);
fprintf('MAE RF Val       : %.2f cycles\n', MAE_val);
fprintf('RMSE Lin Val     : %.2f cycles\n', RMSE_lin_val);
fprintf('Amélioration RF  : %.1f%%\n', 100*(RMSE_lin_val-RMSE_val)/RMSE_lin_val);
fprintf('\n--- Classification ---\n');
fprintf('Accuracy    : %.2f%%\n', Accuracy*100);
fprintf('F1-Score    : %.2f%%\n', F1*100);
fprintf('\n🎉 Analyse optimisée terminée !\n');