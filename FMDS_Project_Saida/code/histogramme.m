%% ==========================================================
%        PROJET FMDS & MAINTENANCE PREDICTIVE
%        Dataset : NASA C-MAPSS (FD001)
%        Author : Saida Zmitri
% ===========================================================

clc; clear; close all;

%% === 1. Charger les données ===============================
% Assurez-vous que 'train_FD001.txt' est dans le dossier '..\data'
data = readtable('..\data\train_FD001.txt', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);
data(:, all(ismissing(data))) = []; % Supprimer les colonnes vides

% Renommer les colonnes selon la documentation NASA
colNames = ["id","cycle","setting1","setting2","setting3", ...
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10", ...
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"];
data.Properties.VariableNames = colNames;

disp('✅ Données chargées avec succès :');
disp(head(data));

%% === 2. Exploration rapide =================================
nEngines = numel(unique(data.id));
fprintf('\nNombre de moteurs : %d\n', nEngines);

% Exemple : évolution d’un capteur pour le moteur 1
engine_id = 1;
subset = data(data.id == engine_id, :);
figure;
plot(subset.cycle, subset.s2, 'LineWidth', 1.5);
xlabel('Cycle'); ylabel('Capteur s2');
title(['Évolution du capteur s2 - Moteur ', num2str(engine_id)]);
grid on;

%% === 3. Analyse FMDS de base ===============================
% Dernier cycle pour chaque moteur (fin de vie)
lastCycle = varfun(@max, data, 'InputVariables', 'cycle', 'GroupingVariables', 'id');
MTBF = mean(lastCycle.max_cycle);
fprintf('\nMTBF estimé : %.2f cycles\n', MTBF);

% Simuler un temps moyen de réparation (MTTR)
MTTR = 5; % cycles
Disponibilite = MTBF / (MTBF + MTTR);
fprintf('Taux de disponibilité estimé : %.2f %%\n', Disponibilite * 100);

% Tracer la fonction de fiabilité (loi de Weibull)
t = 0:1:MTBF;
beta = 2; eta = MTBF; % paramètres Weibull
R = exp(-(t/eta).^beta);
figure;
plot(t, R, 'LineWidth', 2);
xlabel('Temps (cycles)'); ylabel('R(t)');
title('Fonction de Fiabilité - Loi de Weibull');
grid on;

%% === 3b. Histogramme des cycles avant panne ==================
cycles_before_failure = lastCycle.max_cycle;  % Nombre de cycles avant panne par moteur
figure;
histogram(cycles_before_failure, 20, 'FaceColor',[0.2 0.6 0.8], 'EdgeColor','black');
title('Histogramme des cycles avant panne');
xlabel('Nombre de cycles avant panne');
ylabel('Nombre de moteurs');
grid on;

% Statistiques descriptives
fprintf('\nStatistiques des cycles avant panne :\n');
fprintf('Cycles minimum : %.0f\n', min(cycles_before_failure));
fprintf('Cycles maximum : %.0f\n', max(cycles_before_failure));
fprintf('Cycles moyen   : %.2f\n', mean(cycles_before_failure));
fprintf('Cycles médian  : %.2f\n', median(cycles_before_failure));
fprintf('Écart type    : %.2f\n', std(cycles_before_failure));

%% === 4. Préparation pour le pronostic RUL ===================
% Ajouter la durée de vie résiduelle (RUL)
data = innerjoin(data, lastCycle(:, {'id', 'max_cycle'}), 'Keys', 'id');
data.RUL = data.max_cycle - data.cycle;

% Supprimer les colonnes inutiles pour le modèle
X = data{:, 4:26}; % features (capteurs)
Y = data.RUL;      % target (RUL)

%% === 5. Diagnostic : corrélation capteurs ===================
corrMatrix = corr(X);
figure;
imagesc(corrMatrix);
colorbar;
title('Corrélation entre capteurs');
xlabel('Capteurs'); ylabel('Capteurs');

%% === 6. Modèle de régression linéaire (Pronostic simple) ====
fprintf('\nEntraînement du modèle de régression linéaire...\n');
mdl = fitlm(X, Y);
Y_pred = predict(mdl, X);

RMSE = sqrt(mean((Y - Y_pred).^2));
fprintf('RMSE du modèle : %.2f cycles\n', RMSE);

%% === 7. Visualisation du RUL estimé ========================
engine_id = 1;
subset = data(data.id == engine_id, :);
Y_true = subset.RUL;
Y_pred_sub = predict(mdl, subset{:, 4:26});

figure;
plot(subset.cycle, Y_true, 'b-', 'LineWidth', 2);
hold on;
plot(subset.cycle, Y_pred_sub, 'r--', 'LineWidth', 2);
xlabel('Cycle');
ylabel('RUL');
legend('RUL réel', 'RUL prédit');
title(['Comparaison RUL réel vs prédit - Moteur ', num2str(engine_id)]);
grid on;

%% === 8. Synthèse et interprétation ==========================
fprintf('\n===== SYNTHÈSE =====\n');
fprintf('MTBF moyen        : %.2f cycles\n', MTBF);
fprintf('MTTR supposé      : %.2f cycles\n', MTTR);
fprintf('Disponibilité      : %.2f %%\n', Disponibilite * 100);
fprintf('Erreur RMSE (RUL)  : %.2f cycles\n', RMSE);
fprintf('--------------------------\n');
disp('Interprétation :');
disp('Un RMSE faible indique un bon modèle de pronostic.');
disp('La disponibilité élevée (>90%) traduit un bon comportement global du système.');
