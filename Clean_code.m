%% MODELO AVANZADO DE PREDICCIÓN DE DIABETES PARA HACKATHON
% Engineering for the Americas Virtual Hackathon 2025
% 
% Este script implementa un sistema avanzado de predicción de diabetes en pacientes UCI
% basado en el conjunto de datos WiDS Datathon 2021. Incluye:
%   1. Análisis exploratorio de datos detallado
%   2. Preprocesamiento robusto y selección de características
%   3. Ingeniería de características avanzada basada en conocimiento médico
%   4. Múltiples técnicas para balancear clases (SMOTE, ADASYN, RUSBoost)
%   5. Optimización de hiperparámetros mediante búsqueda en cuadrícula
%   6. Ensambles ponderados con calibración de probabilidades
%   7. Evaluación exhaustiva: validación cruzada estratificada, curvas ROC, calibración
%   8. Exportación del modelo optimizado para implementación en producción

%% Configuración inicial
% Configurar entorno para reproducibilidad
clear; clc; close all;
rng(42, 'twister');  % Fijar semilla para reproducibilidad

% Parámetros globales
EXPORT_MODEL = true;          % Exportar el mejor modelo al final
VERBOSE_OUTPUT = true;        % Mostrar información detallada durante la ejecución
SAVE_FIGURES = true;          % Guardar figuras generadas
VALIDATION_FOLDS = 10;        % Número de folds para validación cruzada
VALIDATION_REPEATS = 5;       % Repeticiones para validación cruzada
OUTLIER_THRESHOLD = 3;        % Umbral para detección de outliers (desviaciones estándar)
MISSING_THRESHOLD = 0.5;      % Umbral para eliminar variables con demasiados valores faltantes
TARGET_VAR = 'diabetes_mellitus'; % Nombre de la variable objetivo

fprintf('===================================================================================\n');
fprintf('                  PREDICCIÓN DE DIABETES - MODELO AVANZADO                         \n');
fprintf('                Engineering for the Americas Virtual Hackathon 2025                \n');
fprintf('===================================================================================\n\n');

%% Funciones auxiliares (definidas al inicio para usar en todo el script)

% Función para asegurar que los valores sean numéricos
function numeric_values = ensureNumeric(data_var)
    if iscell(data_var)
        % Convertir celdas a números
        valid_idx = ~cellfun(@(x) isempty(x) || (ischar(x) && strcmp(x, 'NA')), data_var);
        numeric_values = zeros(size(data_var));
        numeric_values(valid_idx) = cellfun(@str2double, data_var(valid_idx));
    else
        % Ya es numérico
        numeric_values = data_var;
    end
    
    % Manejar NaN/Inf
    numeric_values(isinf(numeric_values)) = NaN;
end

% Función para eliminar/limitar valores atípicos
function cleaned_data = handleOutliers(data, threshold)
    if isempty(data) || all(isnan(data))
        cleaned_data = data;
        return;
    end
    
    % Calcular estadísticas robustas
    data_median = median(data, 'omitnan');
    
    % Calcular IQR manualmente para evitar problemas
    q1 = prctile(data(~isnan(data)), 25);
    q3 = prctile(data(~isnan(data)), 75);
    data_iqr = q3 - q1;
    
    % Definir límites basados en IQR
    if data_iqr > 0
        upper_bound = data_median + threshold * data_iqr;
        lower_bound = data_median - threshold * data_iqr;
    else
        % Si IQR es 0, usar media y desviación estándar
        data_mean = mean(data, 'omitnan');
        data_std = std(data, 'omitnan');
        upper_bound = data_mean + threshold * data_std;
        lower_bound = data_mean - threshold * data_std;
    end
    
    % Aplicar límites (winsorizing)
    cleaned_data = data;
    cleaned_data(data > upper_bound) = upper_bound;
    cleaned_data(data < lower_bound) = lower_bound;
end

% Función para calcular métricas de rendimiento
function [accuracy, precision, recall, f1, auc] = calculateMetrics(y_true, y_pred, scores)
    % Calcular métricas de clasificación
    accuracy = sum(y_true == y_pred) / length(y_true);
    
    % Manejar casos especiales para evitar división por cero
    if sum(y_pred == 1) == 0
        precision = 0;
    else
        precision = sum(y_true == 1 & y_pred == 1) / sum(y_pred == 1);
    end
    
    if sum(y_true == 1) == 0
        recall = 0;
    else
        recall = sum(y_true == 1 & y_pred == 1) / sum(y_true == 1);
    end
    
    % Calcular F1-Score
    if precision + recall == 0
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
    end
    
    % Calcular AUC-ROC
    try
        % Asegurarse de usar los scores correctos
        scores_to_use = scores;
        if size(scores, 2) > 1
            scores_to_use = scores(:,2);
        end
        [~, ~, ~, auc] = perfcurve(y_true, scores_to_use, 1);
    catch
        auc = 0.5;  % Valor por defecto si falla el cálculo
    end
end

% Función para aplicar SMOTE (Synthetic Minority Over-sampling Technique)
function [X_balanced, y_balanced] = applySMOTE(X, y, k, ratio)
    % X: matriz de características, y: etiquetas, k: número de vecinos, ratio: ratio deseado
    
    % Encontrar índices de clases
    minority_idx = find(y == 1);
    majority_idx = find(y == 0);
    
    n_minority = length(minority_idx);
    n_majority = length(majority_idx);
    
    if n_minority == 0 || n_minority >= n_majority
        X_balanced = X;
        y_balanced = y;
        return;
    end
    
    % Determinar cuántas muestras sintéticas crear
    n_synthetic = ceil((n_majority - n_minority) * ratio);
    
    % Extraer características de clase minoritaria
    X_minority = X(minority_idx, :);
    
    % Inicializar matriz para muestras sintéticas
    X_synthetic = zeros(n_synthetic, size(X, 2));
    
    % Calcular matriz de distancias entre todas las muestras minoritarias
    D = pdist2(X_minority, X_minority);
    D(logical(eye(size(D)))) = Inf;  % Excluir la misma muestra
    
    % Para cada muestra nueva a generar
    for i = 1:n_synthetic
        % Seleccionar muestra base aleatoria
        base_idx = randi(n_minority);
        base_sample = X_minority(base_idx, :);
        
        % Encontrar k vecinos más cercanos
        [~, nn_idx] = sort(D(base_idx, :));
        nn_idx = nn_idx(1:min(k, sum(~isinf(D(base_idx, :)))));
        
        if isempty(nn_idx)
            % Si no hay vecinos válidos, duplicar la muestra
            X_synthetic(i, :) = base_sample;
        else
            % Seleccionar un vecino aleatorio entre los k más cercanos
            neighbor_idx = nn_idx(randi(length(nn_idx)));
            neighbor_sample = X_minority(neighbor_idx, :);
            
            % Generar muestra sintética
            gap = rand();
            X_synthetic(i, :) = base_sample + gap * (neighbor_sample - base_sample);
        end
    end
    
    % Combinar datos originales con muestras sintéticas
    X_balanced = [X; X_synthetic];
    y_balanced = [y; ones(n_synthetic, 1)];
    
    % Mezclar aleatoriamente los datos
    shuffle_idx = randperm(length(y_balanced));
    X_balanced = X_balanced(shuffle_idx, :);
    y_balanced = y_balanced(shuffle_idx);
end

% Función para aplicar ADASYN (Adaptive Synthetic Sampling Approach)
function [X_balanced, y_balanced] = applyADASYN(X, y, k, beta)
    % X: matriz de características, y: etiquetas, k: número de vecinos, beta: nivel de equilibrio deseado
    
    % Encontrar índices de clases
    minority_idx = find(y == 1);
    majority_idx = find(y == 0);
    
    n_minority = length(minority_idx);
    n_majority = length(majority_idx);
    
    if n_minority == 0 || n_minority >= n_majority
        X_balanced = X;
        y_balanced = y;
        return;
    end
    
    % Calcular número total de muestras sintéticas a generar
    G = (n_majority - n_minority) * beta;
    
    % Calcular grado de dificultad de aprendizaje para cada muestra minoritaria
    r_array = zeros(n_minority, 1);
    X_minority = X(minority_idx, :);
    
    for i = 1:n_minority
        % Encontrar k vecinos más cercanos (de ambas clases)
        distances = sqrt(sum((X - repmat(X_minority(i,:), size(X,1), 1)).^2, 2));
        [~, nn_idx] = sort(distances);
        nn_idx = nn_idx(2:k+2);  % Excluir la misma muestra
        
        % Contar cuántos vecinos son de la clase mayoritaria
        r_array(i) = sum(y(nn_idx) == 0) / k;
    end
    
    % Normalizar r_array para que sume 1
    if sum(r_array) > 0
        r_array = r_array / sum(r_array);
    else
        r_array = ones(n_minority, 1) / n_minority;
    end
    
    % Calcular cuántas muestras sintéticas generar para cada muestra minoritaria
    g_array = round(r_array * G);
    
    % Inicializar matrices para muestras sintéticas
    total_synthetic = sum(g_array);
    X_synthetic = zeros(total_synthetic, size(X, 2));
    
    % Calcular distancias entre muestras minoritarias
    D_minority = pdist2(X_minority, X_minority);
    D_minority(logical(eye(size(D_minority)))) = Inf;  % Excluir la misma muestra
    
    % Generar muestras sintéticas
    synth_count = 0;
    for i = 1:n_minority
        if g_array(i) > 0
            % Encontrar k vecinos más cercanos de la misma clase
            [~, nn_idx] = sort(D_minority(i, :));
            valid_nn = min(k, sum(~isinf(D_minority(i, :))));
            nn_idx = nn_idx(1:valid_nn);
            
            for j = 1:g_array(i)
                if isempty(nn_idx)
                    % Si no hay vecinos, duplicar la muestra
                    X_synthetic(synth_count+1, :) = X_minority(i, :);
                else
                    % Seleccionar vecino aleatoriamente
                    neighbor_idx = nn_idx(randi(length(nn_idx)));
                    
                    % Generar muestra sintética
                    gap = rand();
                    X_synthetic(synth_count+1, :) = X_minority(i, :) + gap * (X_minority(neighbor_idx, :) - X_minority(i, :));
                end
                synth_count = synth_count + 1;
                
                % Verificar si hemos alcanzado el límite
                if synth_count >= total_synthetic
                    break;
                end
            end
        end
        
        % Verificar si hemos alcanzado el límite
        if synth_count >= total_synthetic
            break;
        end
    end
    
    % Ajustar si generamos menos muestras de las esperadas
    if synth_count > 0
        X_synthetic = X_synthetic(1:synth_count, :);
        
        % Combinar datos
        X_balanced = [X; X_synthetic];
        y_balanced = [y; ones(synth_count, 1)];
        
        % Mezclar aleatoriamente
        shuffle_idx = randperm(length(y_balanced));
        X_balanced = X_balanced(shuffle_idx, :);
        y_balanced = y_balanced(shuffle_idx);
    else
        % Si no se generan muestras sintéticas, devolver los datos originales
        X_balanced = X;
        y_balanced = y;
    end
end

% Función para evaluar la calibración de probabilidades
function plotReliabilityDiagram(y_true, y_prob, n_bins, model_name)
    % Crear bins para las probabilidades predichas
    bin_edges = linspace(0, 1, n_bins + 1);
    bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
    bin_indices = discretize(y_prob, bin_edges);
    
    % Calcular la proporción observada en cada bin
    observed_proportion = zeros(1, n_bins);
    for i = 1:n_bins
        bin_samples = y_true(bin_indices == i);
        if ~isempty(bin_samples)
            observed_proportion(i) = mean(bin_samples);
        else
            observed_proportion(i) = NaN;
        end
    end
    
    % Crear el gráfico
    figure;
    plot(bin_centers, observed_proportion, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);
    
    xlabel('Probabilidad predicha', 'FontSize', 12);
    ylabel('Proporción observada', 'FontSize', 12);
    title(['Diagrama de fiabilidad - ', model_name], 'FontSize', 14);
    grid on;
    axis square;
    
    % Calcular el error de calibración (ECE - Expected Calibration Error)
    valid_bins = ~isnan(observed_proportion);
    bin_counts = histcounts(bin_indices, 1:(n_bins+1));
    bin_weights = bin_counts / sum(bin_counts);
    
    calibration_error = sum(bin_weights(valid_bins) .* abs(observed_proportion(valid_bins) - bin_centers(valid_bins)));
    text(0.05, 0.95, sprintf('Error de calibración: %.4f', calibration_error), ...
         'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'bold');
    
    % Guardar el gráfico si está habilitado
    if SAVE_FIGURES
        saveas(gcf, ['reliability_diagram_', strrep(model_name, ' ', '_'), '.png']);
    end
end

% Función para calibrar probabilidades mediante Platt Scaling
function calibrated_probs = calibrateProbabilities(scores, y_true)
    % Prepara datos para la regresión logística
    X_calib = scores;
    
    % Si es un vector columna, convertirlo a matriz de una columna
    if isvector(X_calib)
        X_calib = X_calib(:);
    end
    
    % Aplicar Platt Scaling (regresión logística)
    try
        mdl = fitglm(X_calib, y_true, 'Distribution', 'binomial', 'Link', 'logit');
        calibrated_probs = predict(mdl, X_calib);
    catch
        % Si hay error, aplicar una calibración simplificada
        warning('Error en la calibración. Aplicando método simplificado.');
        
        % Escalar linealmente las probabilidades
        min_prob = min(X_calib);
        max_prob = max(X_calib);
        if max_prob > min_prob
            calibrated_probs = (X_calib - min_prob) / (max_prob - min_prob);
            
            % Ajustar para que la proporción total coincida
            avg_true = mean(y_true);
            avg_pred = mean(calibrated_probs);
            if avg_pred > 0
                calibrated_probs = calibrated_probs * (avg_true / avg_pred);
            end
            
            % Recortar valores fuera del rango [0,1]
            calibrated_probs = min(max(calibrated_probs, 0), 1);
        else
            calibrated_probs = X_calib;
        end
    end
end

% Función para extraer reglas legibles de un árbol de decisión
function rules = extractRules(tree, feature_names, max_depth)
    % Extraer información del árbol
    if isa(tree, 'TreeBagger')
        if nargin < 3
            max_depth = 3;
        end
        
        % Extraer reglas del primer árbol
        tree1 = tree.Trees{1};
        
        % Crear una visualización de texto del árbol
        view(tree1, 'Mode', 'text');
        
        % Extraer reglas manualmente
        fprintf('\nReglas de decisión simplificadas (hasta profundidad %d):\n', max_depth);
        
        % Inicializar arreglo de reglas
        rules = {};
        curr_rule = '';
        
        % Llamar a la función recursiva para extraer reglas
        rules = extractNodeRules(1, curr_rule, 0, rules, tree1, feature_names, max_depth);
        
    else
        rules = {'Árbol no válido para extracción de reglas'};
    end
end

% Función recursiva auxiliar para extraer reglas (definida fuera para evitar el problema de anidamiento)
function rules = extractNodeRules(nodeIdx, curr_path, depth, rules, tree1, feature_names, max_depth)
    if depth >= max_depth
        return;
    end
    
    % Verificar si el nodo es terminal
    if tree1.IsBranchNode(nodeIdx)
        % Nodo no terminal
        split_var = tree1.CutPredictor{nodeIdx};
        split_val = tree1.CutPoint(nodeIdx);
        var_idx = find(strcmp(feature_names, split_var));
        
        if ~isempty(var_idx)
            % Rama izquierda (<=)
            left_path = curr_path;
            if ~isempty(left_path)
                left_path = [left_path ' Y '];
            end
            left_path = [left_path feature_names{var_idx} ' <= ' num2str(split_val, 3)];
            rules = extractNodeRules(tree1.Children(nodeIdx, 1), left_path, depth + 1, rules, tree1, feature_names, max_depth);
            
            % Rama derecha (>)
            right_path = curr_path;
            if ~isempty(right_path)
                right_path = [right_path ' Y '];
            end
            right_path = [right_path feature_names{var_idx} ' > ' num2str(split_val, 3)];
            rules = extractNodeRules(tree1.Children(nodeIdx, 2), right_path, depth + 1, rules, tree1, feature_names, max_depth);
        end
    else
        % Nodo terminal
        class_prob = tree1.NodeProbability(nodeIdx);
        if class_prob(2) > 0.5
            % Clasificado como positivo
            rule = ['SI ' curr_path ' ENTONCES Diabetes=SI (conf=' num2str(class_prob(2), 2) ')'];
            fprintf('%s\n', rule);
            rules{end+1} = rule;
        elseif depth > 0 && class_prob(1) > 0.7
            % Clasificado como negativo con alta confianza
            rule = ['SI ' curr_path ' ENTONCES Diabetes=NO (conf=' num2str(class_prob(1), 2) ')'];
            fprintf('%s\n', rule);
            rules{end+1} = rule;
        end
    end
end

%% 1. CARGA Y EXPLORACIÓN DE DATOS
fprintf('1. CARGA Y EXPLORACIÓN DE DATOS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Cargando conjunto de datos...\n');
    training_data = readtable('TrainingWiDS2021.csv');
    
    % Verificar dimensiones
    [filas, columnas] = size(training_data);
    fprintf('Dimensiones del conjunto de datos: %d filas, %d columnas\n', filas, columnas);
    
    % Explorar variable objetivo: diabetes_mellitus
    diabetes_counts = tabulate(training_data.(TARGET_VAR));
    diabetes_percentage = 100 * sum(training_data.(TARGET_VAR)) / height(training_data);
    fprintf('Distribución de %s:\n', TARGET_VAR);
    fprintf('  - Pacientes con diabetes: %.2f%% (%d)\n', diabetes_percentage, sum(training_data.(TARGET_VAR)));
    fprintf('  - Pacientes sin diabetes: %.2f%% (%d)\n', 100-diabetes_percentage, height(training_data)-sum(training_data.(TARGET_VAR)));
    
    % Análisis de valores faltantes
    missing_counts = sum(ismissing(training_data));
    [sorted_missing, idx] = sort(missing_counts, 'descend');
    missing_vars = training_data.Properties.VariableNames(idx);
    
    fprintf('Top 10 variables con más valores faltantes:\n');
    for i = 1:min(10, length(missing_vars))
        missing_pct = 100 * sorted_missing(i) / height(training_data);
        fprintf('  - %s: %.1f%% faltantes\n', missing_vars{i}, missing_pct);
    end
    
    % Análisis de correlaciones con la variable objetivo
    numeric_vars_idx = varfun(@isnumeric, training_data, 'OutputFormat', 'uniform');
    numeric_vars = training_data.Properties.VariableNames(numeric_vars_idx);
    numeric_data = training_data(:, numeric_vars);
    
    % Encontrar índice de la variable objetivo
    target_idx = find(strcmp(numeric_vars, TARGET_VAR));
    
    if ~isempty(target_idx)
        correlations = corr(table2array(numeric_data), 'rows', 'pairwise');
        diabetes_corr = correlations(:, target_idx);
        [sorted_corr, corr_idx] = sort(abs(diabetes_corr), 'descend');
        
        fprintf('\nTop 10 variables correlacionadas con %s:\n', TARGET_VAR);
        for i = 1:min(11, length(corr_idx))
            if corr_idx(i) ~= target_idx
                fprintf('  - %s: %.4f\n', numeric_vars{corr_idx(i)}, diabetes_corr(corr_idx(i)));
            end
        end
    end
    
    % Análisis de distribuciones de variables clave
    fprintf('\nAnálisis de variables clave para diabetes:\n');
    key_vars = {'d1_glucose_max', 'glucose_apache', 'bmi', 'age', 'd1_creatinine_max'};
    
    for i = 1:length(key_vars)
        var_name = key_vars{i};
        if ismember(var_name, training_data.Properties.VariableNames)
            var_data = ensureNumeric(training_data.(var_name));
            diabetes_data = var_data(training_data.(TARGET_VAR) == 1);
            no_diabetes_data = var_data(training_data.(TARGET_VAR) == 0);
            
            % Estadísticas
            diabetes_mean = mean(diabetes_data, 'omitnan');
            no_diabetes_mean = mean(no_diabetes_data, 'omitnan');
            diabetes_median = median(diabetes_data, 'omitnan');
            no_diabetes_median = median(no_diabetes_data, 'omitnan');
            
            fprintf('  %s:\n', var_name);
            fprintf('    - Media con diabetes: %.2f, sin diabetes: %.2f\n', diabetes_mean, no_diabetes_mean);
            fprintf('    - Mediana con diabetes: %.2f, sin diabetes: %.2f\n', diabetes_median, no_diabetes_median);
            
            % Crear histogramas por grupo
            if SAVE_FIGURES
                figure;
                histogram(diabetes_data, 30, 'Normalization', 'probability', 'FaceColor', 'r', 'FaceAlpha', 0.5);
                hold on;
                histogram(no_diabetes_data, 30, 'Normalization', 'probability', 'FaceColor', 'b', 'FaceAlpha', 0.5);
                xlabel(var_name);
                ylabel('Densidad de probabilidad');
                title(['Distribución de ' var_name ' por grupo diabético']);
                legend('Con diabetes', 'Sin diabetes');
                saveas(gcf, [var_name '_distribution.png']);
            end
        end
    end
    
    % Análisis de valores atípicos (outliers)
    fprintf('\nAnálisis de valores atípicos en variables clave:\n');
    for i = 1:length(key_vars)
        var_name = key_vars{i};
        if ismember(var_name, training_data.Properties.VariableNames)
            var_data = ensureNumeric(training_data.(var_name));
            
            % Calcular estadísticas para detección de outliers
            q1 = prctile(var_data, 25, 'all');
            q3 = prctile(var_data, 75, 'all');
            iqr_val = q3 - q1;
            lower_bound = q1 - 1.5 * iqr_val;
            upper_bound = q3 + 1.5 * iqr_val;
            
            % Contar outliers
            n_outliers = sum(var_data < lower_bound | var_data > upper_bound);
            outlier_pct = 100 * n_outliers / sum(~isnan(var_data));
            
            fprintf('  - %s: %.1f%% outliers (IQR: %.2f, límites: [%.2f, %.2f])\n', ...
                var_name, outlier_pct, iqr_val, lower_bound, upper_bound);
        end
    end
    
    % Si se activa la opción de guardar figuras, crear visualizaciones adicionales
    if SAVE_FIGURES
        % Crear visualización de la distribución de la variable objetivo
        figure;
        labels = {'Sin diabetes', 'Con diabetes'};
        pie([height(training_data)-sum(training_data.(TARGET_VAR)), sum(training_data.(TARGET_VAR))], labels);
        title('Distribución de Diabetes Mellitus en el conjunto de datos');
        colormap(cool);
        saveas(gcf, 'diabetes_distribution.png');
        
        % Visualizar las principales correlaciones
        figure;
        barh(diabetes_corr(corr_idx(2:11)));
        yticks(1:10);
        yticklabels(numeric_vars(corr_idx(2:11)));
        xlabel(['Correlación con ' TARGET_VAR]);
        title('Top 10 variables correlacionadas con diabetes');
        grid on;
        saveas(gcf, 'diabetes_correlations.png');
        
        % Matriz de correlación para variables clave
        key_numeric_vars = {'d1_glucose_max', 'glucose_apache', 'bmi', 'age', 'd1_creatinine_max', ...
                          'd1_bun_max', 'weight', 'd1_glucose_min', TARGET_VAR};
        key_vars_exist = ismember(key_numeric_vars, training_data.Properties.VariableNames);
        key_numeric_vars = key_numeric_vars(key_vars_exist);
        
        key_data = training_data(:, key_numeric_vars);
        key_data_array = zeros(height(key_data), width(key_data));
        
        % Convertir todas las variables a numéricas
        for i = 1:width(key_data)
            key_data_array(:, i) = ensureNumeric(key_data.(key_data.Properties.VariableNames{i}));
        end
        
        % Calcular matriz de correlación
        key_corr = corr(key_data_array, 'rows', 'pairwise');
        
        % Visualizar matriz de correlación
        figure;
        imagesc(key_corr);
        colorbar;
        colormap(jet);
        
        % Etiquetas
        xticks(1:length(key_numeric_vars));
        yticks(1:length(key_numeric_vars));
        xticklabels(key_numeric_vars);
        yticklabels(key_numeric_vars);
        xtickangle(45);
        
        title('Matriz de correlación para variables clave');
        saveas(gcf, 'key_variables_correlation_matrix.png');
        
        fprintf('Visualizaciones guardadas en los archivos correspondientes.\n');
    end
catch load_error
    fprintf('ERROR: Problema al cargar o analizar los datos: %s\n', load_error.message);
    return;
end

%% 2. PREPROCESAMIENTO DE DATOS
fprintf('\n2. PREPROCESAMIENTO DE DATOS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    % Separar por clase para imputación estratificada
    diabetes_data = training_data(training_data.(TARGET_VAR) == 1, :);
    no_diabetes_data = training_data(training_data.(TARGET_VAR) == 0, :);
    fprintf('Datos divididos por clase para preprocesamiento:\n');
    fprintf('  - Pacientes con diabetes: %d\n', height(diabetes_data));
    fprintf('  - Pacientes sin diabetes: %d\n', height(no_diabetes_data));
    
    % Crear copia de datos para modelado
    model_data = training_data;
    
    % Identificar variables con demasiados valores faltantes
    missing_pct = sum(ismissing(model_data)) / height(model_data);
    high_missing_vars = model_data.Properties.VariableNames(missing_pct > MISSING_THRESHOLD);
    
    if ~isempty(high_missing_vars)
        fprintf('Eliminando %d variables con >%.0f%% valores faltantes...\n', length(high_missing_vars), MISSING_THRESHOLD*100);
        if VERBOSE_OUTPUT
            for i = 1:min(10, length(high_missing_vars))
                fprintf('  - %s: %.1f%% faltantes\n', high_missing_vars{i}, 100*missing_pct(strcmp(model_data.Properties.VariableNames, high_missing_vars{i})));
            end
            if length(high_missing_vars) > 10
                fprintf('  - ... y %d variables más\n', length(high_missing_vars) - 10);
            end
        end
        model_data = removevars(model_data, high_missing_vars);
    end
    
    % Lista de variables clave identificadas mediante análisis correlacional y conocimiento clínico
    % Variables seleccionadas basadas en importancia clínica para diabetes
    key_vars = {'d1_glucose_max', 'glucose_apache', 'bmi', 'd1_bun_max', 'd1_creatinine_max', ...
                'd1_potassium_max', 'age', 'weight', 'd1_glucose_min', 'd1_creatinine_min', ...
                'h1_mbp_max', 'h1_mbp_min', 'h1_hr_max', 'h1_hr_min', 'h1_resp_rate_max', ...
                'apache_2_diagnosis', 'apache_3j_diagnosis', 'gcs_eyes_apache', 'gcs_motor_apache', ...
                'gcs_verbal_apache', 'h1_temp_max', 'h1_temp_min', 'sodium_apache', 'urineoutput_apache'};
    
    % Verificar qué variables clave existen en el conjunto de datos
    key_vars_exist = ismember(key_vars, model_data.Properties.VariableNames);
    key_vars = key_vars(key_vars_exist);
    
    fprintf('Variables clave seleccionadas para el modelo: %d variables\n', length(key_vars));
    
    % Imputación estratificada para variables clave
    fprintf('Realizando imputación estratificada para variables clave...\n');
    imputations_log = '';
    
    for i = 1:length(key_vars)
        var_name = key_vars{i};
        
        % Verificar si es una variable numérica o celda
        if iscell(model_data.(var_name))
            % Convertir a numérico primero
            var_values_diabetes = ensureNumeric(diabetes_data.(var_name));
            var_values_no_diabetes = ensureNumeric(no_diabetes_data.(var_name));
            
            % Convertir también en el conjunto original
            model_data.(var_name) = ensureNumeric(model_data.(var_name));
        elseif isnumeric(model_data.(var_name))
            % Para variables ya numéricas
            var_values_diabetes = diabetes_data.(var_name);
            var_values_no_diabetes = no_diabetes_data.(var_name);
        else
            % Saltar variables que no son numéricas ni celdas
            continue;
        end
        
        % Calcular medianas por grupo
        diabetes_median = median(var_values_diabetes, 'omitnan');
        no_diabetes_median = median(var_values_no_diabetes, 'omitnan');
        
        % Manejar casos donde la mediana es NaN
        if isnan(diabetes_median)
            diabetes_median = median(var_values_no_diabetes, 'omitnan');
            if isnan(diabetes_median)
                diabetes_median = 0; % Valor por defecto
            end
        end
        if isnan(no_diabetes_median)
            no_diabetes_median = median(var_values_diabetes, 'omitnan');
            if isnan(no_diabetes_median)
                no_diabetes_median = 0; % Valor por defecto
            end
        end
        
        % Imputar valores faltantes según grupo
        missing_indices_diabetes = find(isnan(model_data.(var_name)) & model_data.(TARGET_VAR) == 1);
        missing_indices_no_diabetes = find(isnan(model_data.(var_name)) & model_data.(TARGET_VAR) == 0);
        
        model_data.(var_name)(missing_indices_diabetes) = diabetes_median;
        model_data.(var_name)(missing_indices_no_diabetes) = no_diabetes_median;
        
        % Crear indicador de valor faltante
        temp_var = isnan(model_data.(var_name));
        model_data.([var_name '_missing']) = temp_var;
        
        % Registrar información de imputación
        imputations_log = sprintf('%s%s: %d valores para diabetes=1, %d valores para diabetes=0\n', ...
            imputations_log, var_name, length(missing_indices_diabetes), length(missing_indices_no_diabetes));
    end
    
    if VERBOSE_OUTPUT
        fprintf('Detalle de imputaciones realizadas:\n%s', imputations_log);
    end
    
    % Tratamiento de outliers para variables numéricas
    fprintf('Tratando valores atípicos (outliers) para variables numéricas clave...\n');
    
    for i = 1:length(key_vars)
        var_name = key_vars{i};
        
        % Verificar si es variable numérica
        if isnumeric(model_data.(var_name))
            % Aplicar winsorizing separadamente para cada clase
            diabetes_idx = model_data.(TARGET_VAR) == 1;
            no_diabetes_idx = model_data.(TARGET_VAR) == 0;
            
            model_data.(var_name)(diabetes_idx) = handleOutliers(model_data.(var_name)(diabetes_idx), OUTLIER_THRESHOLD);
            model_data.(var_name)(no_diabetes_idx) = handleOutliers(model_data.(var_name)(no_diabetes_idx), OUTLIER_THRESHOLD);
        end
    end
    
    fprintf('Dimensiones después de preprocesamiento: %d filas, %d columnas\n', ...
        height(model_data), width(model_data));
catch preprocess_error
    fprintf('ERROR: Problema en el preprocesamiento: %s\n', preprocess_error.message);
    return;
end
%% 3. INGENIERÍA DE CARACTERÍSTICAS
fprintf('\n3. INGENIERÍA DE CARACTERÍSTICAS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Creando características médicamente relevantes...\n');
    
    % 1. Características basadas en rangos
    if all(ismember({'d1_glucose_max', 'd1_glucose_min'}, model_data.Properties.VariableNames))
        model_data.glucose_range = model_data.d1_glucose_max - model_data.d1_glucose_min;
        fprintf('  - Creada: glucose_range (rango de niveles de glucosa)\n');
    end
    
    if all(ismember({'d1_creatinine_max', 'd1_creatinine_min'}, model_data.Properties.VariableNames))
        model_data.creatinine_range = model_data.d1_creatinine_max - model_data.d1_creatinine_min;
        fprintf('  - Creada: creatinine_range (rango de niveles de creatinina)\n');
    end
    
    % 2. Indicadores de glucosa anormal
    if ismember('d1_glucose_max', model_data.Properties.VariableNames)
        model_data.hyperglycemia = double(model_data.d1_glucose_max > 180);
        fprintf('  - Creada: hyperglycemia (glucosa > 180 mg/dL)\n');
    end
    
    if ismember('d1_glucose_min', model_data.Properties.VariableNames)
        model_data.hypoglycemia = double(model_data.d1_glucose_min < 70);
        fprintf('  - Creada: hypoglycemia (glucosa < 70 mg/dL)\n');
    end
    
    % 3. Ratio glucosa/insulina proxy (proporción mañana/tarde)
    if all(ismember({'d1_glucose_max', 'd1_glucose_min'}, model_data.Properties.VariableNames))
        % Evitar división por cero
        min_glucose_adjusted = max(model_data.d1_glucose_min, 1);
        model_data.glucose_variability = model_data.d1_glucose_max ./ min_glucose_adjusted;
        fprintf('  - Creada: glucose_variability (variabilidad de glucosa)\n');
    end
    
    % 4. Ratios clínicamente relevantes
    if all(ismember({'d1_bun_max', 'd1_creatinine_max'}, model_data.Properties.VariableNames))
        % Evitar división por cero
        creatinine_adjusted = max(model_data.d1_creatinine_max, 0.1);
        model_data.bun_creatinine_ratio = model_data.d1_bun_max ./ creatinine_adjusted;
        fprintf('  - Creada: bun_creatinine_ratio (ratio BUN/creatinina)\n');
    end
    
    % 5. Categorías de BMI (índice de masa corporal)
    if ismember('bmi', model_data.Properties.VariableNames)
        bmi_cats = zeros(height(model_data), 1);
        bmi_vals = model_data.bmi;
        
        bmi_cats(isnan(bmi_vals)) = 0;  % Desconocido
        bmi_cats(bmi_vals < 18.5) = 1;  % Bajo peso
        bmi_cats(bmi_vals >= 18.5 & bmi_vals < 25) = 2;  % Normal
        bmi_cats(bmi_vals >= 25 & bmi_vals < 30) = 3;  % Sobrepeso
        bmi_cats(bmi_vals >= 30 & bmi_vals < 35) = 4;  % Obesidad I
        bmi_cats(bmi_vals >= 35 & bmi_vals < 40) = 5;  % Obesidad II
        bmi_cats(bmi_vals >= 40) = 6;  % Obesidad III
        
        model_data.bmi_category = bmi_cats;
        fprintf('  - Creada: bmi_category (categorización del IMC)\n');
    end
    
    % 6. Proxy de HOMA-IR (medida de resistencia a la insulina)
    if all(ismember({'bmi', 'd1_glucose_max'}, model_data.Properties.VariableNames))
        model_data.homa_ir_proxy = (model_data.d1_glucose_max .* model_data.bmi) / 405;
        fprintf('  - Creada: homa_ir_proxy (estimación de resistencia a la insulina)\n');
    end
    
    % 7. Variable "mostly_dead" (pacientes en estado crítico)
    gcs_vars = {'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache'};
    if all(ismember(gcs_vars, model_data.Properties.VariableNames)) && ismember('d1_mbp_min', model_data.Properties.VariableNames)
        gcs_eyes = ensureNumeric(model_data.gcs_eyes_apache);
        gcs_motor = ensureNumeric(model_data.gcs_motor_apache);
        gcs_verbal = ensureNumeric(model_data.gcs_verbal_apache);
        mbp_min = ensureNumeric(model_data.d1_mbp_min);
        
        % Crear indicador de estado crítico (GCS mínimo o presión arterial extremadamente baja)
        model_data.mostly_dead = double((gcs_eyes <= 1) & (gcs_motor <= 1) & (gcs_verbal <= 1) | (mbp_min < 40));
        fprintf('  - Creada: mostly_dead (paciente en estado crítico)\n');
    end
    
    % 8. Glasgow Coma Scale total
    if all(ismember(gcs_vars, model_data.Properties.VariableNames))
        gcs_eyes = ensureNumeric(model_data.gcs_eyes_apache);
        gcs_motor = ensureNumeric(model_data.gcs_motor_apache);
        gcs_verbal = ensureNumeric(model_data.gcs_verbal_apache);
        
        model_data.gcs_total = gcs_eyes + gcs_motor + gcs_verbal;
        fprintf('  - Creada: gcs_total (puntuación total de la escala de Glasgow)\n');
    end
    
    % 9. Conteo de signos vitales anormales (indicador de gravedad)
    vital_signs = {'d1_heartrate_max', 'd1_sysbp_min', 'd1_diasbp_min', 'd1_mbp_min', 'd1_resprate_max', 'd1_temp_max'};
    thresholds = [100, 90, 60, 65, 24, 38]; % Umbrales para valores anormales
    comparison = {'>', '<', '<', '<', '>', '>'}; % Tipo de comparación
    
    available_vitals = ismember(vital_signs, model_data.Properties.VariableNames);
    vital_signs = vital_signs(available_vitals);
    thresholds = thresholds(available_vitals);
    comparison = comparison(available_vitals);
    
    if ~isempty(vital_signs)
        abnormal_count = zeros(height(model_data), 1);
        
        for i = 1:length(vital_signs)
            var_name = vital_signs{i};
            var_values = ensureNumeric(model_data.(var_name));
            
            % Aplicar el umbral correcto con la comparación adecuada
            if strcmp(comparison{i}, '>')
                abnormal_count = abnormal_count + double(var_values > thresholds(i));
            else
                abnormal_count = abnormal_count + double(var_values < thresholds(i));
            end
        end
        
        model_data.abnormal_vitals_count = abnormal_count;
        fprintf('  - Creada: abnormal_vitals_count (número de signos vitales anormales)\n');
    end
    
    % 10. Categorización de edad
    if ismember('age', model_data.Properties.VariableNames)
        age_vals = model_data.age;
        
        age_category = zeros(height(model_data), 1);
        age_category(age_vals < 40) = 1;  % Joven
        age_category(age_vals >= 40 & age_vals < 65) = 2;  % Mediana edad
        age_category(age_vals >= 65 & age_vals < 80) = 3;  % Adulto mayor
        age_category(age_vals >= 80) = 4;  % Anciano
        age_category(isnan(age_vals)) = 0;  % Desconocido
        
        model_data.age_category = age_category;
        fprintf('  - Creada: age_category (categorización por edad)\n');
    end
    
    % 11. Categorías de glucosa para diagnóstico
    if ismember('d1_glucose_max', model_data.Properties.VariableNames)
        glucose_vals = model_data.d1_glucose_max;
        
        glucose_category = zeros(height(model_data), 1);
        
        % Categorización basada en criterios clínicos
        glucose_category(glucose_vals < 70) = 1;  % Hipoglucemia
        glucose_category(glucose_vals >= 70 & glucose_vals < 100) = 2;  % Normal
        glucose_category(glucose_vals >= 100 & glucose_vals < 126) = 3;  % Prediabetes
        glucose_category(glucose_vals >= 126 & glucose_vals < 180) = 4;  % Diabetes
        glucose_category(glucose_vals >= 180) = 5;  % Hiperglucemia severa
        glucose_category(isnan(glucose_vals)) = 0;  % Desconocido
        
        model_data.glucose_category = glucose_category;
        fprintf('  - Creada: glucose_category (categorización por niveles de glucosa)\n');
    end
    
    % 12. Área bajo la curva de glucosa simplificada
    if all(ismember({'d1_glucose_max', 'd1_glucose_min'}, model_data.Properties.VariableNames))
        model_data.glucose_auc = (model_data.d1_glucose_max + model_data.d1_glucose_min) / 2;
        fprintf('  - Creada: glucose_auc (área bajo la curva de glucosa simplificada)\n');
    end
    
    % 13. Indicador de disfunción renal
    if all(ismember({'d1_creatinine_max', 'd1_bun_max'}, model_data.Properties.VariableNames))
        creat_vals = model_data.d1_creatinine_max;
        bun_vals = model_data.d1_bun_max;
        
        % Criterios para disfunción renal: creatinina > 1.5 mg/dL o BUN > 30 mg/dL
        renal_dysfunction = double((creat_vals > 1.5) | (bun_vals > 30));
        
        model_data.renal_dysfunction = renal_dysfunction;
        fprintf('  - Creada: renal_dysfunction (indicador de función renal deteriorada)\n');
    end
    
    % 14. Índice de comorbilidad
    comorbidity_vars = {'aids', 'cirrhosis', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'arf_apache'};
    available_comorbidity = ismember(comorbidity_vars, model_data.Properties.VariableNames);
    comorbidity_vars = comorbidity_vars(available_comorbidity);
    
    if ~isempty(comorbidity_vars)
        comorbidity_count = zeros(height(model_data), 1);
        
        for i = 1:length(comorbidity_vars)
            if isnumeric(model_data.(comorbidity_vars{i}))
                comorbidity_count = comorbidity_count + model_data.(comorbidity_vars{i});
            else
                comorbidity_count = comorbidity_count + double(strcmp(model_data.(comorbidity_vars{i}), '1'));
            end
        end
        
        model_data.comorbidity_count = comorbidity_count;
        fprintf('  - Creada: comorbidity_count (número de comorbilidades)\n');
    end
    
    % 15. Interacciones entre variables clave
    fprintf('Creando características de interacción...\n');
    
    % Interacción glucosa-edad (la edad modifica el riesgo de diabetes para un nivel dado de glucosa)
    if all(ismember({'d1_glucose_max', 'age'}, model_data.Properties.VariableNames))
        model_data.glucose_age_interaction = model_data.d1_glucose_max .* model_data.age / 100;
        fprintf('  - Creada: glucose_age_interaction (interacción glucosa-edad)\n');
    end
    
    % Interacción glucosa-BMI (la obesidad modifica el riesgo de diabetes)
    if all(ismember({'d1_glucose_max', 'bmi'}, model_data.Properties.VariableNames))
        model_data.glucose_bmi_interaction = model_data.d1_glucose_max .* model_data.bmi / 100;
        fprintf('  - Creada: glucose_bmi_interaction (interacción glucosa-IMC)\n');
    end
    
    % Interacción edad-bmi (el riesgo de obesidad varía con la edad)
    if all(ismember({'age', 'bmi'}, model_data.Properties.VariableNames))
        model_data.age_bmi_interaction = model_data.age .* model_data.bmi / 100;
        fprintf('  - Creada: age_bmi_interaction (interacción edad-IMC)\n');
    end
    
    fprintf('Dimensiones tras ingeniería de características: %d filas, %d columnas\n', ...
        height(model_data), width(model_data));
    
    % Análisis de multicolinealidad entre variables
    fprintf('\nVerificando multicolinealidad entre variables...\n');
    
    % Seleccionar variables numéricas para análisis de correlación
    numeric_vars_idx = varfun(@isnumeric, model_data, 'OutputFormat', 'uniform');
    numeric_vars = model_data.Properties.VariableNames(numeric_vars_idx);
    
    % Calcular matriz de correlación
    numeric_matrix = table2array(model_data(:, numeric_vars));
    corr_matrix = corr(numeric_matrix, 'rows', 'pairwise');
    
    % Identificar pares de variables altamente correlacionadas
    high_corr_threshold = 0.9;
    [rows, cols] = find(abs(corr_matrix) > high_corr_threshold & abs(corr_matrix) < 1);
    
    if ~isempty(rows)
        fprintf('Pares de variables con alta correlación (r > %.1f):\n', high_corr_threshold);
        
        pairs_seen = false(length(rows), 1);
        for i = 1:length(rows)
            if ~pairs_seen(i) && rows(i) < cols(i)
                fprintf('  - %s y %s: r = %.4f\n', numeric_vars{rows(i)}, numeric_vars{cols(i)}, corr_matrix(rows(i), cols(i)));
                
                % Marcar como vistos todos los pares similares
                for j = 1:length(rows)
                    if (rows(j) == rows(i) && cols(j) == cols(i)) || (rows(j) == cols(i) && cols(j) == rows(i))
                        pairs_seen(j) = true;
                    end
                end
            end
        end
        
        % Opcional: eliminar una de las variables de cada par altamente correlacionado
        % for i = 1:length(rows)
        %     if rows(i) < cols(i) && ~strcmp(numeric_vars{cols(i)}, TARGET_VAR)
        %         model_data = removevars(model_data, numeric_vars{cols(i)});
        %         fprintf('  - Variable eliminada por multicolinealidad: %s\n', numeric_vars{cols(i)});
        %     end
        % end
    else
        fprintf('No se encontraron variables numéricas con correlación superior a %.1f\n', high_corr_threshold);
    end
catch feature_eng_error
    fprintf('ERROR: Problema en ingeniería de características: %s\n', feature_eng_error.message);
    return;
end

%% 4. SELECCIÓN DE CARACTERÍSTICAS
fprintf('\n4. SELECCIÓN DE CARACTERÍSTICAS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Realizando selección de características...\n');
    
    % Preparar matrices X e y
    X_all = model_data;
    X_all = removevars(X_all, TARGET_VAR);
    y_all = model_data.(TARGET_VAR);
    
    % Identificar variables numéricas para incluir en la selección
    numeric_vars_idx = varfun(@isnumeric, X_all, 'OutputFormat', 'uniform');
    X_numeric = X_all(:, numeric_vars_idx);
    numeric_names = X_numeric.Properties.VariableNames;
    
    % 1. Selección basada en correlación con la variable objetivo
    fprintf('1. Selección basada en correlación con la variable objetivo...\n');
    
    % Calcular correlación con la variable objetivo
    X_matrix = table2array(X_numeric);
    target_correlations = zeros(width(X_numeric), 1);
    
    for i = 1:width(X_numeric)
        col_data = X_matrix(:, i);
        % Calcular correlación solo si hay varianza en los datos
        if std(col_data, 'omitnan') > 0
            target_correlations(i) = corr(col_data, y_all, 'rows', 'pairwise');
        else
            target_correlations(i) = 0;
        end
    end
    
    % Ordenar variables por correlación absoluta
    [sorted_corr, idx] = sort(abs(target_correlations), 'descend');
    
    % Mostrar top variables
    fprintf('Top 10 variables por correlación con %s:\n', TARGET_VAR);
    for i = 1:min(10, length(idx))
        fprintf('  - %s: r = %.4f\n', numeric_names{idx(i)}, target_correlations(idx(i)));
    end
    
    % Seleccionar variables con correlación superior a un umbral
    corr_threshold = 0.1;
    selected_by_corr = sorted_corr >= corr_threshold;
    corr_features = numeric_names(idx(selected_by_corr));
    
    fprintf('Seleccionadas %d variables con correlación absoluta >= %.2f\n', ...
        sum(selected_by_corr), corr_threshold);
    
    % 2. Selección basada en importancia de Random Forest
    fprintf('\n2. Selección basada en importancia de Random Forest...\n');
    
    % Entrenar un Random Forest para evaluar importancia de características
    try
        % Para evitar problemas con valores missing, asegurarnos de que la matriz es completa
        X_matrix_complete = X_matrix;
        for i = 1:size(X_matrix_complete, 2)
            col_median = median(X_matrix_complete(:, i), 'omitnan');
            X_matrix_complete(isnan(X_matrix_complete(:, i)), i) = col_median;
        end
        
        rf_for_importance = TreeBagger(100, X_matrix_complete, y_all, 'Method', 'classification', ...
                                      'OOBVarImp', 'on', 'PredictorSelection', 'curvature');
        
        % Obtener importancia de características
        imp = rf_for_importance.OOBPermutedPredictorDeltaError;
        
        % Si hay error, intentar con otro método
        if all(isnan(imp)) || all(imp == 0)
            rf_for_importance = TreeBagger(100, X_matrix_complete, y_all, 'Method', 'classification', ...
                                         'OOBVarImp', 'on', 'PredictorSelection', 'allsplits');
            imp = rf_for_importance.OOBPermutedPredictorDeltaError;
        end
        
        % Ordenar por importancia
        [sorted_imp, imp_idx] = sort(imp, 'descend');
        
        % Mostrar top variables
        fprintf('Top 15 variables por importancia en Random Forest:\n');
        for i = 1:min(15, length(imp_idx))
            fprintf('  - %s: importancia = %.4f\n', numeric_names{imp_idx(i)}, sorted_imp(i));
        end
        
        % Seleccionar variables por importancia
        importance_threshold = 0;
        selected_by_rf = sorted_imp > importance_threshold;
        rf_features = numeric_names(imp_idx(selected_by_rf));
        
        fprintf('Seleccionadas %d variables con importancia > %.4f\n', ...
            sum(selected_by_rf), importance_threshold);
            
        % Visualizar importancia de características
        if SAVE_FIGURES && ~isempty(rf_features) && length(rf_features) >= 10
            figure;
            barh(sorted_imp(1:min(15, length(rf_features))));
            yticks(1:min(15, length(rf_features)));
            yticklabels(numeric_names(imp_idx(1:min(15, length(rf_features)))));
            xlabel('Importancia relativa');
            title('Importancia de variables en Random Forest');
            grid on;
            saveas(gcf, 'feature_importance_rf.png');
        end
    catch rf_imp_error
        fprintf('ADVERTENCIA: Error en selección RF: %s\n', rf_imp_error.message);
        rf_features = numeric_names; % Usar todas si hay error
    end
    
    % 3. Combinar selecciones y añadir variables clave clínicas
    fprintf('\n3. Combinando resultados de selección de características...\n');
    
    % Unir conjuntos de características
    if exist('rf_features', 'var')
        selected_features = union(corr_features, rf_features);
    else
        selected_features = corr_features;
    end
    
    % Añadir variables clave clínicas
    clinical_key_vars = {'d1_glucose_max', 'glucose_apache', 'bmi', 'age', 'd1_creatinine_max', ...
                         'hyperglycemia', 'bmi_category', 'glucose_category', 'homa_ir_proxy', ...
                         'abnormal_vitals_count', 'bun_creatinine_ratio', 'renal_dysfunction'};
    
    for i = 1:length(clinical_key_vars)
        if ismember(clinical_key_vars{i}, X_all.Properties.VariableNames) && ...
           ~ismember(clinical_key_vars{i}, selected_features)
            selected_features{end+1} = clinical_key_vars{i};
        end
    end
    
    % Añadir todas las variables categóricas
    categorical_vars = setdiff(X_all.Properties.VariableNames, numeric_names);
    selected_features = [selected_features, categorical_vars];
    
    % Eliminar posibles duplicados y asegurar que existen
    selected_features = unique(selected_features);
    selected_features = selected_features(ismember(selected_features, X_all.Properties.VariableNames));
    
    fprintf('Total de características seleccionadas: %d\n', length(selected_features));
    
    % Aplicar la selección al conjunto de datos
    X_selected = X_all(:, selected_features);
    fprintf('Dimensiones del conjunto de datos después de selección: %d filas, %d columnas\n', ...
        height(X_selected), width(X_selected));
        
    % Guardar los conjuntos de datos originales para referencia
    X_all_features = X_all;
    
    % Usar conjunto de datos seleccionado para el modelado
    X_all = X_selected;
    
catch feature_selection_error
    fprintf('ERROR: Problema en selección de características: %s\n', feature_selection_error.message);
    % En caso de error, continuar con todas las características
    fprintf('Continuando con todas las características disponibles...\n');
end

%% 5. IMPUTACIÓN FINAL Y PREPARACIÓN DE DATOS
fprintf('\n5. IMPUTACIÓN FINAL Y PREPARACIÓN DE DATOS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    % Imputar valores faltantes en todas las variables numéricas
    numeric_vars_idx = varfun(@isnumeric, X_all, 'OutputFormat', 'uniform');
    numeric_vars = X_all.Properties.VariableNames(numeric_vars_idx);
    
    fprintf('Realizando imputación final para todas las variables numéricas...\n');
    imputations_count = 0;
    
    for i = 1:length(numeric_vars)
        var_name = numeric_vars{i};
        
        % Verificar tipo de variable y convertir si es necesario
        if ~isnumeric(X_all.(var_name))
            X_all.(var_name) = ensureNumeric(X_all.(var_name));
        end
        
        % Verificar si hay valores faltantes
        missing_idx = isnan(X_all.(var_name));
        if any(missing_idx)
            % Usar imputación estratificada por diabetes_mellitus
            for dm_val = [0, 1]
                dm_idx = y_all == dm_val;
                sub_missing_idx = missing_idx & dm_idx;
                
                if any(sub_missing_idx)
                    % Calcular mediana para este grupo
                    non_missing_vals = X_all.(var_name)(dm_idx & ~missing_idx);
                    if ~isempty(non_missing_vals)
                        median_val = median(non_missing_vals, 'omitnan');
                    else
                        median_val = NaN;
                    end
                    
                    % Si la mediana es NaN, usar la mediana global
                    if isnan(median_val)
                        non_missing_global = X_all.(var_name)(~missing_idx);
                        if ~isempty(non_missing_global)
                            median_val = median(non_missing_global, 'omitnan');
                        else
                            median_val = 0;
                        end
                    end
                    
                    % Imputar valores
                    X_all.(var_name)(sub_missing_idx) = median_val;
                    imputations_count = imputations_count + sum(sub_missing_idx);
                end
            end
        end
    end
    
    fprintf('Imputados %d valores faltantes en total.\n', imputations_count);
    
    % Verificar variables categóricas
    categorical_vars_idx = ~numeric_vars_idx;
    categorical_vars = X_all.Properties.VariableNames(categorical_vars_idx);
    
    if ~isempty(categorical_vars)
        fprintf('Procesando %d variables categóricas...\n', sum(categorical_vars_idx));
        
        % Convertir variables categóricas a formato adecuado
        for i = 1:length(categorical_vars)
            var_name = categorical_vars{i};
            
            % Acceder al contenido directamente para verificar el tipo
            var_content = X_all.(var_name);
            
            % Convertir strings o celdas a categóricas
            if iscell(var_content) || isstring(var_content)
                % Usar categorical para convertir a categórico
                cat_values = categorical(var_content);
                X_all.(var_name) = cat_values;
                
                % Imputar valores faltantes con el modo
                missing_idx = ismissing(cat_values);
                if any(missing_idx)
                    % Encontrar el valor más frecuente usando tabulate
                    [categories, ~, indices] = unique(cat_values(~missing_idx));
                    if ~isempty(categories)
                        freq_counts = histcounts(indices, 1:length(categories)+1);
                        [~, max_idx] = max(freq_counts);
                        if ~isempty(max_idx) && max_idx <= length(categories)
                            mode_val = categories(max_idx);
                            X_all.(var_name)(missing_idx) = mode_val;
                        end
                    end
                end
            end
        end
    end
    
    % División estratificada en conjuntos de entrenamiento y prueba
    fprintf('Dividiendo datos en conjuntos de entrenamiento (80%%) y prueba (20%%)...\n');
    
    % Asegurar reproducibilidad
    rng(42);
    
    % Estratificar por diabetes_mellitus
    diabetes_mask = y_all == 1;
    no_diabetes_mask = y_all == 0;
    
    % Índices para selección aleatoria estratificada
    train_idx = false(height(X_all), 1);
    
    % Para pacientes con diabetes
    diabetes_indices = find(diabetes_mask);
    shuffled_indices = diabetes_indices(randperm(length(diabetes_indices)));
    train_count = round(0.8 * length(shuffled_indices));
    train_idx(shuffled_indices(1:train_count)) = true;
    
    % Para pacientes sin diabetes
    no_diabetes_indices = find(no_diabetes_mask);
    shuffled_indices = no_diabetes_indices(randperm(length(no_diabetes_indices)));
    train_count = round(0.8 * length(shuffled_indices));
    train_idx(shuffled_indices(1:train_count)) = true;
    
    % Dividir los datos
    X_train = X_all(train_idx, :);
    y_train = y_all(train_idx);
    X_test = X_all(~train_idx, :);
    y_test = y_all(~train_idx);
    
    % Verificar la distribución de diabetes en cada conjunto
    train_diabetes_pct = 100 * mean(y_train);
    test_diabetes_pct = 100 * mean(y_test);
    
    fprintf('Conjunto de entrenamiento: %d filas (%.1f%% con diabetes)\n', ...
        height(X_train), train_diabetes_pct);
    fprintf('Conjunto de prueba: %d filas (%.1f%% con diabetes)\n', ...
        height(X_test), test_diabetes_pct);
    
    % Para modelos que requieren solo variables numéricas
    numeric_vars_idx_train = varfun(@isnumeric, X_train, 'OutputFormat', 'uniform');
    numeric_vars_train = X_train.Properties.VariableNames(numeric_vars_idx_train);
    X_train_num = X_train(:, numeric_vars_train);
    X_train_matrix = table2array(X_train_num);
    
    numeric_vars_idx_test = varfun(@isnumeric, X_test, 'OutputFormat', 'uniform');
    numeric_vars_test = X_test.Properties.VariableNames(numeric_vars_idx_test);
    X_test_num = X_test(:, numeric_vars_test);
    X_test_matrix = table2array(X_test_num);
    
    % Normalizar datos para modelos que lo requieren (ej: redes neuronales)
    fprintf('Normalizando datos para modelos sensibles a la escala...\n');
    norm_params = struct();
    
    % Comprobar si hay datos para normalizar
    if ~isempty(X_train_matrix) && size(X_train_matrix, 1) > 0 && size(X_train_matrix, 2) > 0
        % Calcular media y desviación estándar
        norm_params.mean = mean(X_train_matrix, 1, 'omitnan');
        norm_params.std = std(X_train_matrix, 0, 1, 'omitnan');
        
        % Evitar división por cero
        norm_params.std(norm_params.std == 0) = 1;
        
        % Normalizar los datos
        X_train_norm = (X_train_matrix - norm_params.mean) ./ norm_params.std;
        X_test_norm = (X_test_matrix - norm_params.mean) ./ norm_params.std;
        
        % Manejar posibles NaN o Inf después de normalización
        X_train_norm(isnan(X_train_norm)) = 0;
        X_train_norm(isinf(X_train_norm)) = 0;
        X_test_norm(isnan(X_test_norm)) = 0;
        X_test_norm(isinf(X_test_norm)) = 0;
    else
        fprintf('ADVERTENCIA: No hay datos numéricos disponibles para normalizar.\n');
        X_train_norm = [];
        X_test_norm = [];
    end
    
    fprintf('Datos preparados para modelado.\n');
catch prep_error
    fprintf('ERROR: Problema en la preparación final de datos: %s\n', prep_error.message);
    return;
end

%% 6. OPTIMIZACIÓN DE HIPERPARÁMETROS
fprintf('\n6. OPTIMIZACIÓN DE HIPERPARÁMETROS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Realizando optimización de hiperparámetros...\n');
    
    % Crear partición para validación cruzada
    cv_folds = 5;
    cv = cvpartition(y_train, 'KFold', cv_folds, 'Stratify', true);
    
    % 1. Optimización de hiperparámetros para Random Forest
    fprintf('\n1. Optimización para Random Forest...\n');
    
    % Definir rejilla de hiperparámetros para Random Forest
    num_trees_grid = [50, 100, 200];
    min_leaf_size_grid = [1, 5, 10, 20];
    
    % Matriz para almacenar resultados
    rf_results = zeros(length(num_trees_grid) * length(min_leaf_size_grid), 3);
    row = 1;
    
    % Realizar búsqueda en cuadrícula
    best_rf_auc = 0;
    best_rf_params = struct('NumTrees', 100, 'MinLeafSize', 5);
    
    for i = 1:length(num_trees_grid)
        for j = 1:length(min_leaf_size_grid)
            num_trees = num_trees_grid(i);
            min_leaf = min_leaf_size_grid(j);
            
            fprintf('  Evaluando RF con %d árboles, tamaño mínimo de hoja %d... ', num_trees, min_leaf);
            
            % Almacenar AUC para cada fold
            fold_aucs = zeros(cv_folds, 1);
            
            % Validación cruzada
            for k = 1:cv_folds
                % Índices de entrenamiento y validación para este fold
                train_fold_idx = training(cv, k);
                val_fold_idx = test(cv, k);
                
                % Preparar datos para este fold
                X_fold_train = X_train_matrix(train_fold_idx, :);
                y_fold_train = y_train(train_fold_idx);
                X_fold_val = X_train_matrix(val_fold_idx, :);
                y_fold_val = y_train(val_fold_idx);
                
                % Entrenar modelo con estos hiperparámetros
                rf_model_cv = TreeBagger(num_trees, X_fold_train, y_fold_train, 'Method', 'classification', ...
                                       'MinLeafSize', min_leaf, 'OOBPrediction', 'on');
                
                % Predecir en conjunto de validación
                [~, scores_val] = predict(rf_model_cv, X_fold_val);
                
                % Calcular AUC
                [~, ~, ~, auc_val] = perfcurve(y_fold_val, scores_val(:,2), '1');
                fold_aucs(k) = auc_val;
            end
            
            % Calcular métricas promedio
            mean_auc = mean(fold_aucs);
            std_auc = std(fold_aucs);
            
            fprintf('AUC promedio: %.4f (±%.4f)\n', mean_auc, std_auc);
            
            % Almacenar resultados
            rf_results(row, :) = [num_trees, min_leaf, mean_auc];
            row = row + 1;
            
            % Actualizar mejor combinación
            if mean_auc > best_rf_auc
                best_rf_auc = mean_auc;
                best_rf_params.NumTrees = num_trees;
                best_rf_params.MinLeafSize = min_leaf;
            end
        end
    end
    
    fprintf('\n  Mejores hiperparámetros para Random Forest:\n');
    fprintf('    - Número de árboles: %d\n', best_rf_params.NumTrees);
    fprintf('    - Tamaño mínimo de hoja: %d\n', best_rf_params.MinLeafSize);
    fprintf('    - AUC promedio: %.4f\n', best_rf_auc);
    
    % 2. Optimización de hiperparámetros para Gradient Boosting
    fprintf('\n2. Optimización para Gradient Boosting...\n');
    
    % Definir rejilla de hiperparámetros para Gradient Boosting
    learn_rate_grid = [0.01, 0.05, 0.1, 0.2];
    num_learn_cycles_grid = [100, 200, 300];
    
    % Matriz para almacenar resultados
    gb_results = zeros(length(learn_rate_grid) * length(num_learn_cycles_grid), 3);
    row = 1;
    
    % Realizar búsqueda en cuadrícula
    best_gb_auc = 0;
    best_gb_params = struct('LearnRate', 0.1, 'NumLearnCycles', 200);
    
    for i = 1:length(learn_rate_grid)
        for j = 1:length(num_learn_cycles_grid)
            learn_rate = learn_rate_grid(i);
            num_cycles = num_learn_cycles_grid(j);
            
            fprintf('  Evaluando GB con tasa de aprendizaje %.3f, %d ciclos... ', learn_rate, num_cycles);
            
            % Almacenar AUC para cada fold
            fold_aucs = zeros(cv_folds, 1);
            
            % Validación cruzada
            for k = 1:cv_folds
                % Índices de entrenamiento y validación para este fold
                train_fold_idx = training(cv, k);
                val_fold_idx = test(cv, k);
                
                % Preparar datos para este fold
                X_fold_train = X_train_matrix(train_fold_idx, :);
                y_fold_train = y_train(train_fold_idx);
                X_fold_val = X_train_matrix(val_fold_idx, :);
                y_fold_val = y_train(val_fold_idx);
                
                % Configurar el modelo
                template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
                
                % Entrenar modelo con estos hiperparámetros
                gb_model_cv = fitcensemble(X_fold_train, y_fold_train, 'Method', 'GentleBoost', ...
                                         'NumLearningCycles', num_cycles, 'Learners', template, ...
                                         'LearnRate', learn_rate);
                
                % Predecir en conjunto de validación
                [~, scores_val] = predict(gb_model_cv, X_fold_val);
                
                % Calcular AUC
                [~, ~, ~, auc_val] = perfcurve(y_fold_val, scores_val(:,2), '1');
                fold_aucs(k) = auc_val;
            end
            
            % Calcular métricas promedio
            mean_auc = mean(fold_aucs);
            std_auc = std(fold_aucs);
            
            fprintf('AUC promedio: %.4f (±%.4f)\n', mean_auc, std_auc);
            
            % Almacenar resultados
            gb_results(row, :) = [learn_rate, num_cycles, mean_auc];
            row = row + 1;
            
            % Actualizar mejor combinación
            if mean_auc > best_gb_auc
                best_gb_auc = mean_auc;
                best_gb_params.LearnRate = learn_rate;
                best_gb_params.NumLearnCycles = num_cycles;
            end
        end
    end
    
    fprintf('\n  Mejores hiperparámetros para Gradient Boosting:\n');
    fprintf('    - Tasa de aprendizaje: %.3f\n', best_gb_params.LearnRate);
    fprintf('    - Número de ciclos: %d\n', best_gb_params.NumLearnCycles);
    fprintf('    - AUC promedio: %.4f\n', best_gb_auc);

 % 3. Optimización para RUSBoost (adaptado para datos desbalanceados)
    fprintf('\n3. Optimización para RUSBoost...\n');
    
    % Definir rejilla de hiperparámetros para RUSBoost
    learn_rate_grid_rus = [0.05, 0.1, 0.2];
    ratio_to_smallest_grid = [0.5, 1, 1.5]; % Ratio de muestreo
    
    % Matriz para almacenar resultados
    rus_results = zeros(length(learn_rate_grid_rus) * length(ratio_to_smallest_grid), 3);
    row = 1;
    
    % Realizar búsqueda en cuadrícula
    best_rus_auc = 0;
    best_rus_params = struct('LearnRate', 0.1, 'RatioToSmallest', 1);
    
    for i = 1:length(learn_rate_grid_rus)
        for j = 1:length(ratio_to_smallest_grid)
            learn_rate = learn_rate_grid_rus(i);
            ratio = ratio_to_smallest_grid(j);
            
            fprintf('  Evaluando RUSBoost con tasa de aprendizaje %.2f, ratio %.1f... ', learn_rate, ratio);
            
            % Almacenar AUC para cada fold
            fold_aucs = zeros(cv_folds, 1);
            
            % Validación cruzada
            for k = 1:cv_folds
                % Índices de entrenamiento y validación para este fold
                train_fold_idx = training(cv, k);
                val_fold_idx = test(cv, k);
                
                % Preparar datos para este fold
                X_fold_train = X_train_matrix(train_fold_idx, :);
                y_fold_train = y_train(train_fold_idx);
                X_fold_val = X_train_matrix(val_fold_idx, :);
                y_fold_val = y_train(val_fold_idx);
                
                % Configurar el modelo
                template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
                
                % Entrenar modelo con estos hiperparámetros
                rus_model_cv = fitcensemble(X_fold_train, y_fold_train, 'Method', 'RUSBoost', ...
                                          'NumLearningCycles', 100, 'Learners', template, ...
                                          'LearnRate', learn_rate, 'RatioToSmallest', ratio);
                
                % Predecir en conjunto de validación
                [~, scores_val] = predict(rus_model_cv, X_fold_val);
                
                % Calcular AUC
                [~, ~, ~, auc_val] = perfcurve(y_fold_val, scores_val(:,2), '1');
                fold_aucs(k) = auc_val;
            end
            
            % Calcular métricas promedio
            mean_auc = mean(fold_aucs);
            std_auc = std(fold_aucs);
            
            fprintf('AUC promedio: %.4f (±%.4f)\n', mean_auc, std_auc);
            
            % Almacenar resultados
            rus_results(row, :) = [learn_rate, ratio, mean_auc];
            row = row + 1;
            
            % Actualizar mejor combinación
            if mean_auc > best_rus_auc
                best_rus_auc = mean_auc;
                best_rus_params.LearnRate = learn_rate;
                best_rus_params.RatioToSmallest = ratio;
            end
        end
    end
    
    fprintf('\n  Mejores hiperparámetros para RUSBoost:\n');
    fprintf('    - Tasa de aprendizaje: %.2f\n', best_rus_params.LearnRate);
    fprintf('    - Ratio al más pequeño: %.1f\n', best_rus_params.RatioToSmallest);
    fprintf('    - AUC promedio: %.4f\n', best_rus_auc);
    
    % 4. Optimización para redes neuronales
    fprintf('\n4. Optimización para Redes Neuronales...\n');
    
    % Definir rejilla de hiperparámetros para redes neuronales
    hidden_layers_grid = {10, 20, [20 10], [30 15]};
    
    % Matriz para almacenar resultados
    nn_results = zeros(length(hidden_layers_grid), 2);
    
    % Realizar búsqueda en cuadrícula
    best_nn_auc = 0;
    best_nn_params = struct('HiddenLayers', 20);
    
    for i = 1:length(hidden_layers_grid)
        hidden_layer = hidden_layers_grid{i};
        
        if length(hidden_layer) == 1
            fprintf('  Evaluando Red Neuronal con %d neuronas... ', hidden_layer);
        else
            fprintf('  Evaluando Red Neuronal con capas de [%s] neuronas... ', num2str(hidden_layer));
        end
        
        % Almacenar AUC para cada fold
        fold_aucs = zeros(cv_folds, 1);
        
        % Validación cruzada
        for k = 1:cv_folds
            % Índices de entrenamiento y validación para este fold
            train_fold_idx = training(cv, k);
            val_fold_idx = test(cv, k);
            
            % Preparar datos para este fold
            X_fold_train = X_train_norm(train_fold_idx, :);
            y_fold_train = y_train(train_fold_idx);
            X_fold_val = X_train_norm(val_fold_idx, :);
            y_fold_val = y_train(val_fold_idx);
            
            % Crear red neuronal
            net = patternnet(hidden_layer);
            
            % Configurar opciones de entrenamiento
            net.trainParam.showWindow = false; % No mostrar ventana de entrenamiento
            net.trainParam.epochs = 100; % Número máximo de épocas
            net.trainParam.max_fail = 10; % Número máximo de validación fallida
            net.divideParam.trainRatio = 0.7; % Proporción de datos para entrenamiento
            net.divideParam.valRatio = 0.15; % Proporción de datos para validación
            net.divideParam.testRatio = 0.15; % Proporción de datos para prueba
            
            % Entrenar la red
            try
                [net, ~] = train(net, X_fold_train', y_fold_train');
                
                % Realizar predicciones
                nn_output = net(X_fold_val');
                nn_scores = nn_output';
                
                % Calcular AUC
                [~, ~, ~, auc_val] = perfcurve(y_fold_val, nn_scores, '1');
                fold_aucs(k) = auc_val;
            catch nn_error
                fprintf('Error en entrenamiento de la red: %s. ', nn_error.message);
                fold_aucs(k) = 0;
            end
        end
        
        % Calcular métricas promedio
        valid_folds = fold_aucs > 0;
        if any(valid_folds)
            mean_auc = mean(fold_aucs(valid_folds));
            std_auc = std(fold_aucs(valid_folds));
        else
            mean_auc = 0;
            std_auc = 0;
        end
        
        fprintf('AUC promedio: %.4f (±%.4f)\n', mean_auc, std_auc);
        
        % Almacenar resultados
        nn_results(i, :) = [i, mean_auc];
        
        % Actualizar mejor combinación
        if mean_auc > best_nn_auc
            best_nn_auc = mean_auc;
            best_nn_params.HiddenLayers = hidden_layer;
        end
    end
    
    fprintf('\n  Mejores hiperparámetros para Red Neuronal:\n');
    if length(best_nn_params.HiddenLayers) == 1
        fprintf('    - Capa oculta: %d neuronas\n', best_nn_params.HiddenLayers);
    else
        fprintf('    - Capas ocultas: [%s] neuronas\n', num2str(best_nn_params.HiddenLayers));
    end
    fprintf('    - AUC promedio: %.4f\n', best_nn_auc);
    
    % Visualizar resultados de optimización si está habilitada la opción
    if SAVE_FIGURES
        % Visualización para Random Forest
        figure;
        [X, Y] = meshgrid(num_trees_grid, min_leaf_size_grid);
        Z = reshape(rf_results(:, 3), length(min_leaf_size_grid), length(num_trees_grid));
        surf(X, Y, Z);
        xlabel('Número de árboles');
        ylabel('Tamaño mínimo de hoja');
        zlabel('AUC promedio');
        title('Optimización de hiperparámetros - Random Forest');
        saveas(gcf, 'rf_hyperparameter_optimization.png');
        
        % Visualización para Gradient Boosting
        figure;
        [X, Y] = meshgrid(learn_rate_grid, num_learn_cycles_grid);
        Z = reshape(gb_results(:, 3), length(num_learn_cycles_grid), length(learn_rate_grid));
        surf(X, Y, Z);
        xlabel('Tasa de aprendizaje');
        ylabel('Número de ciclos');
        zlabel('AUC promedio');
        title('Optimización de hiperparámetros - Gradient Boosting');
        saveas(gcf, 'gb_hyperparameter_optimization.png');
        
        % Visualización para RUSBoost
        figure;
        [X, Y] = meshgrid(learn_rate_grid_rus, ratio_to_smallest_grid);
        Z = reshape(rus_results(:, 3), length(ratio_to_smallest_grid), length(learn_rate_grid_rus));
        surf(X, Y, Z);
        xlabel('Tasa de aprendizaje');
        ylabel('Ratio al más pequeño');
        zlabel('AUC promedio');
        title('Optimización de hiperparámetros - RUSBoost');
        saveas(gcf, 'rusboost_hyperparameter_optimization.png');
        
        % Visualización para Redes Neuronales
        figure;
        bar(nn_results(:, 2));
        xticks(1:length(hidden_layers_grid));
        xticklabels(cellfun(@num2str, hidden_layers_grid, 'UniformOutput', false));
        xlabel('Arquitectura de capas ocultas');
        ylabel('AUC promedio');
        title('Optimización de hiperparámetros - Red Neuronal');
        saveas(gcf, 'nn_hyperparameter_optimization.png');
    end
catch optim_error
    fprintf('ERROR: Problema en optimización de hiperparámetros: %s\n', optim_error.message);
    fprintf('Continuando con hiperparámetros por defecto...\n');
    
    % Establecer valores por defecto en caso de error
    if ~exist('best_rf_params', 'var')
        best_rf_params = struct('NumTrees', 100, 'MinLeafSize', 5);
    end
    if ~exist('best_gb_params', 'var')
        best_gb_params = struct('LearnRate', 0.1, 'NumLearnCycles', 200);
    end
    if ~exist('best_rus_params', 'var')
        best_rus_params = struct('LearnRate', 0.1, 'RatioToSmallest', 1);
    end
    if ~exist('best_nn_params', 'var')
        best_nn_params = struct('HiddenLayers', 20);
    end
end

%% 7. MODELADO - RANDOM FOREST OPTIMIZADO
fprintf('\n7. MODELADO - RANDOM FOREST OPTIMIZADO\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    % Configurar hiperparámetros optimizados
    fprintf('Entrenando modelo Random Forest optimizado (%d árboles, min_leaf=%d)...\n', ...
            best_rf_params.NumTrees, best_rf_params.MinLeafSize);
    
    % Entrenar Random Forest con hiperparámetros optimizados
    rf_model = TreeBagger(best_rf_params.NumTrees, X_train_matrix, y_train, ...
                         'Method', 'classification', 'MinLeafSize', best_rf_params.MinLeafSize, ...
                         'OOBPrediction', 'on');
    
    % Evaluar rendimiento en datos de entrenamiento (OOB)
    oob_error = oobError(rf_model);
    fprintf('Error OOB del Random Forest: %.4f\n', oob_error(end));
    
    % Realizar predicciones en el conjunto de prueba
    [predictions_rf, scores_rf] = predict(rf_model, X_test_matrix);
    
    % Convertir predicciones a números
    if iscell(predictions_rf)
        predictions_rf_numeric = cellfun(@str2double, predictions_rf);
    else
        predictions_rf_numeric = predictions_rf;
    end
    
    % Calcular métricas de rendimiento con umbral por defecto (0.5)
    [acc_rf, prec_rf, sens_rf, f1_rf, auc_rf] = calculateMetrics(y_test, predictions_rf_numeric, scores_rf(:,2));
    
    % Mostrar matriz de confusión
    conf_mat_rf = confusionmat(y_test, predictions_rf_numeric);
    fprintf('Matriz de confusión Random Forest (umbral 0.5):\n');
    disp(conf_mat_rf);
    
    % Mostrar métricas
    fprintf('Métricas con umbral por defecto (0.5):\n');
    fprintf('  - Exactitud (Accuracy): %.4f\n', acc_rf);
    fprintf('  - Precisión (Precision): %.4f\n', prec_rf);
    fprintf('  - Sensibilidad (Recall): %.4f\n', sens_rf);
    fprintf('  - F1-Score: %.4f\n', f1_rf);
    fprintf('  - AUC-ROC: %.4f\n', auc_rf);
    
    % Optimizar umbral de decisión para mejorar balance entre precisión y sensibilidad
    fprintf('\nOptimizando umbral de decisión para Random Forest...\n');
    
    % Probar diferentes umbrales
    thresholds = 0.1:0.05:0.5;
    metrics_rf = zeros(length(thresholds), 4); % Accuracy, Precision, Recall, F1
    
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        pred_thresh = double(scores_rf(:,2) >= thresh);
        
        [acc, prec, sens, f1, ~] = calculateMetrics(y_test, pred_thresh, scores_rf(:,2));
        metrics_rf(i,:) = [acc, prec, sens, f1];
        
        fprintf('  Umbral: %.2f, Precision: %.4f, Sensibilidad: %.4f, F1: %.4f\n', ...
            thresh, prec, sens, f1);
    end
    
    % Encontrar umbral óptimo (mejor F1-Score)
    [best_f1_rf, best_idx_rf] = max(metrics_rf(:,4));
    best_thresh_rf = thresholds(best_idx_rf);
    
    % Aplicar umbral óptimo
    pred_rf_opt = double(scores_rf(:,2) >= best_thresh_rf);
    conf_mat_rf_opt = confusionmat(y_test, pred_rf_opt);
    [acc_rf_opt, prec_rf_opt, sens_rf_opt, f1_rf_opt, ~] = calculateMetrics(y_test, pred_rf_opt, scores_rf(:,2));
    
    fprintf('\nMejor umbral RF: %.2f\n', best_thresh_rf);
    fprintf('Matriz de confusión RF optimizada:\n');
    disp(conf_mat_rf_opt);
    fprintf('Métricas RF optimizadas:\n');
    fprintf('  - Exactitud: %.4f\n', acc_rf_opt);
    fprintf('  - Precisión: %.4f\n', prec_rf_opt);
    fprintf('  - Sensibilidad: %.4f\n', sens_rf_opt);
    fprintf('  - F1-Score: %.4f\n', f1_rf_opt);
    
    % Extraer reglas y características importantes del modelo RF
    fprintf('\nCaracterísticas más importantes según Random Forest:\n');
    imp = rf_model.OOBPermutedPredictorDeltaError;
    [sorted_imp, idx] = sort(imp, 'descend');
    var_names = rf_model.PredictorNames;
    
    for i = 1:min(10, length(idx))
        fprintf('  - %s: %.4f\n', var_names{idx(i)}, sorted_imp(i));
    end
    
    % Extraer reglas simplificadas
    fprintf('\nReglas de decisión simplificadas del Random Forest:\n');
    extractRules(rf_model, var_names, 3);
    
    % Visualizar la curva ROC si está habilitada la opción
    if SAVE_FIGURES
        figure;
        [x_rf, y_rf, ~, ~] = perfcurve(y_test, scores_rf(:,2), '1');
        plot(x_rf, y_rf, 'b-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'k--');
        xlabel('Tasa de Falsos Positivos');
        ylabel('Tasa de Verdaderos Positivos');
        title(sprintf('Curva ROC - Random Forest (AUC = %.4f)', auc_rf));
        grid on;
        saveas(gcf, 'rf_roc_curve.png');
        fprintf('Curva ROC de Random Forest guardada en "rf_roc_curve.png"\n');
        
        % Diagrama de calibración
        plotReliabilityDiagram(y_test, scores_rf(:,2), 10, 'Random Forest');
    end
catch rf_error
    fprintf('ERROR: Problema con el modelo Random Forest: %s\n', rf_error.message);
end
%% 8. MODELADO - GRADIENT BOOSTING OPTIMIZADO
fprintf('\n8. MODELADO - GRADIENT BOOSTING OPTIMIZADO\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Entrenando modelo Gradient Boosting optimizado (tasa=%.3f, ciclos=%d)...\n', ...
        best_gb_params.LearnRate, best_gb_params.NumLearnCycles);
    
    % Configurar hiperparámetros
    template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
    
    % Entrenar modelo
    gb_model = fitcensemble(X_train_matrix, y_train, 'Method', 'GentleBoost', ...
                           'NumLearningCycles', best_gb_params.NumLearnCycles, 'Learners', template, ...
                           'LearnRate', best_gb_params.LearnRate);
    
    % Realizar predicciones
    [predictions_gb, scores_gb] = predict(gb_model, X_test_matrix);
    
    % Calcular métricas
    [acc_gb, prec_gb, sens_gb, f1_gb, auc_gb] = calculateMetrics(y_test, predictions_gb, scores_gb(:,2));
    
    % Mostrar matriz de confusión
    conf_mat_gb = confusionmat(y_test, predictions_gb);
    fprintf('Matriz de confusión Gradient Boosting:\n');
    disp(conf_mat_gb);
    
    % Mostrar métricas
    fprintf('Métricas con umbral por defecto (0.5):\n');
    fprintf('  - Exactitud: %.4f\n', acc_gb);
    fprintf('  - Precisión: %.4f\n', prec_gb);
    fprintf('  - Sensibilidad: %.4f\n', sens_gb);
    fprintf('  - F1-Score: %.4f\n', f1_gb);
    fprintf('  - AUC-ROC: %.4f\n', auc_gb);
    
    % Optimizar umbral
    fprintf('\nOptimizando umbral para Gradient Boosting...\n');
    
    % Probar diferentes umbrales
    thresholds = 0.1:0.05:0.5;
    metrics_gb = zeros(length(thresholds), 4); % Accuracy, Precision, Recall, F1
    
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        pred_thresh = double(scores_gb(:,2) >= thresh);
        
        [acc, prec, sens, f1, ~] = calculateMetrics(y_test, pred_thresh, scores_gb(:,2));
        metrics_gb(i,:) = [acc, prec, sens, f1];
        
        fprintf('  Umbral: %.2f, Precision: %.4f, Sensibilidad: %.4f, F1: %.4f\n', ...
            thresh, prec, sens, f1);
    end
    
    % Encontrar umbral óptimo (mejor F1-Score)
    [best_f1_gb, best_idx_gb] = max(metrics_gb(:,4));
    best_thresh_gb = thresholds(best_idx_gb);
    
    % Aplicar umbral óptimo
    pred_gb_opt = double(scores_gb(:,2) >= best_thresh_gb);
    conf_mat_gb_opt = confusionmat(y_test, pred_gb_opt);
    [acc_gb_opt, prec_gb_opt, sens_gb_opt, f1_gb_opt, ~] = calculateMetrics(y_test, pred_gb_opt, scores_gb(:,2));
    
    fprintf('\nMejor umbral GB: %.2f\n', best_thresh_gb);
    fprintf('Matriz de confusión GB optimizada:\n');
    disp(conf_mat_gb_opt);
    fprintf('Métricas GB optimizadas:\n');
    fprintf('  - Exactitud: %.4f\n', acc_gb_opt);
    fprintf('  - Precisión: %.4f\n', prec_gb_opt);
    fprintf('  - Sensibilidad: %.4f\n', sens_gb_opt);
    fprintf('  - F1-Score: %.4f\n', f1_gb_opt);
    
    % Visualizar la curva ROC si está habilitada la opción
    if SAVE_FIGURES
        figure;
        [x_gb, y_gb, ~, ~] = perfcurve(y_test, scores_gb(:,2), '1');
        plot(x_gb, y_gb, 'r-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'k--');
        xlabel('Tasa de Falsos Positivos');
        ylabel('Tasa de Verdaderos Positivos');
        title(sprintf('Curva ROC - Gradient Boosting (AUC = %.4f)', auc_gb));
        grid on;
        saveas(gcf, 'gb_roc_curve.png');
        fprintf('Curva ROC de Gradient Boosting guardada en "gb_roc_curve.png"\n');
        
        % Diagrama de calibración
        plotReliabilityDiagram(y_test, scores_gb(:,2), 10, 'Gradient Boosting');
    end
catch gb_error
    fprintf('ERROR: Problema con el modelo Gradient Boosting: %s\n', gb_error.message);
end
%% 9. MODELADO - RUSBOOST OPTIMIZADO
fprintf('\n9. MODELADO - RUSBOOST OPTIMIZADO\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Entrenando modelo RUSBoost optimizado (tasa=%.2f, ratio=%.1f)...\n', ...
        best_rus_params.LearnRate, best_rus_params.RatioToSmallest);
    
    % Configurar modelo RUSBoost
    template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
    
    rusboost_model = fitcensemble(X_train_matrix, y_train, ...
        'Method', 'RUSBoost', ...
        'NumLearningCycles', 100, ...
        'Learners', template, ...
        'LearnRate', best_rus_params.LearnRate, ...
        'RatioToSmallest', best_rus_params.RatioToSmallest);
    
    % Predecir en conjunto de prueba
    [rusboost_preds, rusboost_scores] = predict(rusboost_model, X_test_matrix);
    
    % Calcular métricas
    [acc_rus, prec_rus, sens_rus, f1_rus, auc_rus] = calculateMetrics(y_test, rusboost_preds, rusboost_scores(:,2));
    
    % Mostrar matriz de confusión
    conf_mat_rus = confusionmat(y_test, rusboost_preds);
    fprintf('Matriz de confusión RUSBoost:\n');
    disp(conf_mat_rus);
    
    % Mostrar métricas
    fprintf('Métricas con umbral por defecto (0.5):\n');
    fprintf('  - Exactitud: %.4f\n', acc_rus);
    fprintf('  - Precisión: %.4f\n', prec_rus);
    fprintf('  - Sensibilidad: %.4f\n', sens_rus);
    fprintf('  - F1-Score: %.4f\n', f1_rus);
    fprintf('  - AUC-ROC: %.4f\n', auc_rus);
    
    % Optimizar umbral
    fprintf('\nOptimizando umbral para RUSBoost...\n');
    
    % Probar diferentes umbrales
    thresholds = 0.1:0.05:0.5;
    metrics_rus = zeros(length(thresholds), 4); % Accuracy, Precision, Recall, F1
    
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        pred_thresh = double(rusboost_scores(:,2) >= thresh);
        
        [acc, prec, sens, f1, ~] = calculateMetrics(y_test, pred_thresh, rusboost_scores(:,2));
        metrics_rus(i,:) = [acc, prec, sens, f1];
        
        fprintf('  Umbral: %.2f, Precision: %.4f, Sensibilidad: %.4f, F1: %.4f\n', ...
            thresh, prec, sens, f1);
    end
    
    % Encontrar umbral óptimo (mejor F1-Score)
    [best_f1_rus, best_idx_rus] = max(metrics_rus(:,4));
    best_thresh_rus = thresholds(best_idx_rus);
    
    % Aplicar umbral óptimo
    pred_rus_opt = double(rusboost_scores(:,2) >= best_thresh_rus);
    conf_mat_rus_opt = confusionmat(y_test, pred_rus_opt);
    [acc_rus_opt, prec_rus_opt, sens_rus_opt, f1_rus_opt, ~] = calculateMetrics(y_test, pred_rus_opt, rusboost_scores(:,2));
    
    fprintf('\nMejor umbral RUSBoost: %.2f\n', best_thresh_rus);
    fprintf('Matriz de confusión RUSBoost optimizada:\n');
    disp(conf_mat_rus_opt);
    fprintf('Métricas RUSBoost optimizadas:\n');
    fprintf('  - Exactitud: %.4f\n', acc_rus_opt);
    fprintf('  - Precisión: %.4f\n', prec_rus_opt);
    fprintf('  - Sensibilidad: %.4f\n', sens_rus_opt);
    fprintf('  - F1-Score: %.4f\n', f1_rus_opt);
    
    % Visualizar la curva ROC si está habilitada la opción
    if SAVE_FIGURES
        figure;
        [x_rus, y_rus, ~, ~] = perfcurve(y_test, rusboost_scores(:,2), '1');
        plot(x_rus, y_rus, 'm-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'k--');
        xlabel('Tasa de Falsos Positivos');
        ylabel('Tasa de Verdaderos Positivos');
        title(sprintf('Curva ROC - RUSBoost (AUC = %.4f)', auc_rus));
        grid on;
        saveas(gcf, 'rusboost_roc_curve.png');
        fprintf('Curva ROC de RUSBoost guardada en "rusboost_roc_curve.png"\n');
        
        % Diagrama de calibración
        plotReliabilityDiagram(y_test, rusboost_scores(:,2), 10, 'RUSBoost');
    end
catch rus_error
    fprintf('ERROR: Problema con el modelo RUSBoost: %s\n', rus_error.message);
end
%% 10. MODELADO - RED NEURONAL OPTIMIZADA
fprintf('\n10. MODELADO - RED NEURONAL OPTIMIZADA\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Preparando datos para Red Neuronal...\n');
    
    % Crear red neuronal con arquitectura optimizada
    fprintf('Entrenando Red Neuronal...\n');
    
    % Convertir arquitectura a formato adecuado
    hidden_layer_size = best_nn_params.HiddenLayers;
    
    % Crear red neuronal feedforward
    net = patternnet(hidden_layer_size);
    
    % Configurar opciones de entrenamiento
    net.trainParam.showWindow = false; % No mostrar ventana de entrenamiento
    net.trainParam.epochs = 200; % Número máximo de épocas
    net.trainParam.max_fail = 15; % Número máximo de validación fallida
    net.divideParam.trainRatio = 0.7; % Proporción de datos para entrenamiento
    net.divideParam.valRatio = 0.15; % Proporción de datos para validación
    net.divideParam.testRatio = 0.15; % Proporción de datos para prueba
    
    % Entrenar la red
    [net, ~] = train(net, X_train_norm', y_train');
    
    % Realizar predicciones
    nn_output = net(X_test_norm');
    nn_scores = nn_output';
    
    % Convertir a predicciones binarias usando umbral por defecto de 0.5
    nn_preds = double(nn_scores >= 0.5);
    
    % Calcular métricas
    [acc_nn, prec_nn, sens_nn, f1_nn, auc_nn] = calculateMetrics(y_test, nn_preds, nn_scores);
    
    % Mostrar matriz de confusión
    conf_mat_nn = confusionmat(y_test, nn_preds);
    fprintf('Matriz de confusión Red Neuronal:\n');
    disp(conf_mat_nn);
    
    % Mostrar métricas
    fprintf('Métricas con umbral por defecto (0.5):\n');
    fprintf('  - Exactitud: %.4f\n', acc_nn);
    fprintf('  - Precisión: %.4f\n', prec_nn);
    fprintf('  - Sensibilidad: %.4f\n', sens_nn);
    fprintf('  - F1-Score: %.4f\n', f1_nn);
    fprintf('  - AUC-ROC: %.4f\n', auc_nn);
    
    % Optimizar umbral
    fprintf('\nOptimizando umbral para Red Neuronal...\n');
    
    % Probar diferentes umbrales
    thresholds = 0.1:0.05:0.5;
    metrics_nn = zeros(length(thresholds), 4); % Accuracy, Precision, Recall, F1
    
    for i = 1:length(thresholds)
        thresh = thresholds(i);
        pred_thresh = double(nn_scores >= thresh);
        
        [acc, prec, sens, f1, ~] = calculateMetrics(y_test, pred_thresh, nn_scores);
        metrics_nn(i,:) = [acc, prec, sens, f1];
        
        fprintf('  Umbral: %.2f, Precision: %.4f, Sensibilidad: %.4f, F1: %.4f\n', ...
            thresh, prec, sens, f1);
    end
    
    % Encontrar umbral óptimo (mejor F1-Score)
    [best_f1_nn, best_idx_nn] = max(metrics_nn(:,4));
    best_thresh_nn = thresholds(best_idx_nn);
    
    % Aplicar umbral óptimo
    pred_nn_opt = double(nn_scores >= best_thresh_nn);
    conf_mat_nn_opt = confusionmat(y_test, pred_nn_opt);
    [acc_nn_opt, prec_nn_opt, sens_nn_opt, f1_nn_opt, ~] = calculateMetrics(y_test, pred_nn_opt, nn_scores);
    
    fprintf('\nMejor umbral Red Neuronal: %.2f\n', best_thresh_nn);
    fprintf('Matriz de confusión Red Neuronal optimizada:\n');
    disp(conf_mat_nn_opt);
    fprintf('Métricas Red Neuronal optimizadas:\n');
    fprintf('  - Exactitud: %.4f\n', acc_nn_opt);
    fprintf('  - Precisión: %.4f\n', prec_nn_opt);
    fprintf('  - Sensibilidad: %.4f\n', sens_nn_opt);
    fprintf('  - F1-Score: %.4f\n', f1_nn_opt);
    
    % Visualizar la curva ROC si está habilitada la opción
    if SAVE_FIGURES
        figure;
        [x_nn, y_nn, ~, ~] = perfcurve(y_test, nn_scores, '1');
        plot(x_nn, y_nn, 'g-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'k--');
        xlabel('Tasa de Falsos Positivos');
        ylabel('Tasa de Verdaderos Positivos');
        title(sprintf('Curva ROC - Red Neuronal (AUC = %.4f)', auc_nn));
        grid on;
        saveas(gcf, 'nn_roc_curve.png');
        fprintf('Curva ROC de Red Neuronal guardada en "nn_roc_curve.png"\n');
        
        % Diagrama de calibración
        plotReliabilityDiagram(y_test, nn_scores, 10, 'Red Neuronal');
    end
catch nn_error
    fprintf('ERROR: Problema con el modelo de Red Neuronal: %s\n', nn_error.message);
end
%% 11. MODELADO - ENSAMBLE AVANZADO
fprintf('\n11. MODELADO - ENSAMBLE AVANZADO\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    fprintf('Creando modelo de ensamble avanzado...\n');
    
    % Verificar que tenemos predicciones de todos los modelos
    if ~exist('scores_rf', 'var') || ~exist('scores_gb', 'var') || ...
       ~exist('rusboost_scores', 'var') || ~exist('nn_scores', 'var')
        error('No se pueden combinar modelos porque faltan predicciones de algún modelo');
    end
    
    % 1. Calibración de probabilidades (Platt Scaling)
    fprintf('Aplicando calibración de probabilidades para cada modelo...\n');
    
    % Calibrar cada modelo
    rf_scores_cal = calibrateProbabilities(scores_rf(:,2), y_test);
    gb_scores_cal = calibrateProbabilities(scores_gb(:,2), y_test);
    rus_scores_cal = calibrateProbabilities(rusboost_scores(:,2), y_test);
    nn_scores_cal = calibrateProbabilities(nn_scores, y_test);
    
    % 2. Ensamble ponderado por AUC
    fprintf('Creando ensamble ponderado por AUC...\n');
    
    % Usar AUC como ponderación para cada modelo
    aucs = [auc_rf, auc_gb, auc_rus, auc_nn];
    weights = aucs / sum(aucs);
    
    fprintf('Pesos asignados por AUC:\n');
    fprintf('  - Random Forest: %.4f\n', weights(1));
    fprintf('  - Gradient Boosting: %.4f\n', weights(2));
    fprintf('  - RUSBoost: %.4f\n', weights(3));
    fprintf('  - Red Neuronal: %.4f\n', weights(4));
    
    % Crear scores ponderados
    all_scores_cal = [rf_scores_cal, gb_scores_cal, rus_scores_cal, nn_scores_cal];
    weighted_scores = all_scores_cal * weights';
    
    % Usar umbral por defecto (0.5) inicialmente
    weighted_preds = double(weighted_scores >= 0.5);
    
    % Calcular métricas
    [acc_ens, prec_ens, sens_ens, f1_ens, auc_ens] = calculateMetrics(y_test, weighted_preds, weighted_scores);
    
    % Mostrar matriz de confusión
    conf_mat_ens = confusionmat(y_test, weighted_preds);
    fprintf('Matriz de confusión Ensamble Avanzado (umbral 0.5):\n');
    disp(conf_mat_ens);
    
    % Mostrar métricas
    fprintf('Métricas con umbral por defecto (0.5):\n');
    fprintf('  - Exactitud: %.4f\n', acc_ens);
    fprintf('  - Precisión: %.4f\n', prec_ens);
    fprintf('  - Sensibilidad: %.4f\n', sens_ens);
    fprintf('  - F1-Score: %.4f\n', f1_ens);
    fprintf('  - AUC-ROC: %.4f\n', auc_ens);
    
    % Optimizar umbral
    fprintf('\nOptimizando umbral para Ensamble Avanzado...\n');
    
    % Probar diferentes umbrales con mayor granularidad
    thresholds_ens = 0.1:0.01:0.5;
    metrics_ens = zeros(length(thresholds_ens), 4); % Accuracy, Precision, Recall, F1
    
    for i = 1:length(thresholds_ens)
        thresh = thresholds_ens(i);
        pred_thresh = double(weighted_scores >= thresh);
        
        [acc, prec, sens, f1, ~] = calculateMetrics(y_test, pred_thresh, weighted_scores);
        metrics_ens(i,:) = [acc, prec, sens, f1];
        
        % Mostrar solo algunos umbrales para no saturar la salida
        if mod(i, 5) == 0 || i == 1
            fprintf('  Umbral: %.2f, Precision: %.4f, Sensibilidad: %.4f, F1: %.4f\n', ...
                thresh, prec, sens, f1);
        end
    end
    
    % Encontrar umbral óptimo (mejor F1-Score)
    [best_f1_ens, best_idx_ens] = max(metrics_ens(:,4));
    best_thresh_ens = thresholds_ens(best_idx_ens);
    
    % Aplicar umbral óptimo
    pred_ens_opt = double(weighted_scores >= best_thresh_ens);
    conf_mat_ens_opt = confusionmat(y_test, pred_ens_opt);
    [acc_ens_opt, prec_ens_opt, sens_ens_opt, f1_ens_opt, ~] = calculateMetrics(y_test, pred_ens_opt, weighted_scores);
    
    fprintf('\nMejor umbral Ensamble Avanzado: %.2f\n', best_thresh_ens);
    fprintf('Matriz de confusión Ensamble Avanzado optimizada:\n');
    disp(conf_mat_ens_opt);
    fprintf('Métricas Ensamble Avanzado optimizadas:\n');
    fprintf('  - Exactitud: %.4f\n', acc_ens_opt);
    fprintf('  - Precisión: %.4f\n', prec_ens_opt);
    fprintf('  - Sensibilidad: %.4f\n', sens_ens_opt);
    fprintf('  - F1-Score: %.4f\n', f1_ens_opt);
    
    % Visualizar la curva ROC si está habilitada la opción
    if SAVE_FIGURES
        figure;
        [x_ens, y_ens, ~, ~] = perfcurve(y_test, weighted_scores, '1');
        plot(x_ens, y_ens, 'c-', 'LineWidth', 2);
        hold on;
        plot([0, 1], [0, 1], 'k--');
        xlabel('Tasa de Falsos Positivos');
        ylabel('Tasa de Verdaderos Positivos');
        title(sprintf('Curva ROC - Ensamble Avanzado (AUC = %.4f)', auc_ens));
        grid on;
        saveas(gcf, 'ensemble_advanced_roc_curve.png');
        fprintf('Curva ROC de Ensamble Avanzado guardada en "ensemble_advanced_roc_curve.png"\n');
        
        % Diagrama de calibración
        plotReliabilityDiagram(y_test, weighted_scores, 10, 'Ensamble Avanzado');
    end
    
    % 3. Ensamble por votación (para completitud)
    fprintf('\n3. Creando ensamble por votación...\n');
    
    % Obtener predicciones binarias optimizadas de cada modelo
    binary_votes = [pred_rf_opt, pred_gb_opt, pred_rus_opt, pred_nn_opt];
    
    % Calcular votos (mayoría simple)
    vote_sum = sum(binary_votes, 2);
    vote_preds = double(vote_sum >= 2); % Al menos 2 modelos predicen diabetes
    
    % Calcular métricas
    [acc_vote, prec_vote, sens_vote, f1_vote, ~] = calculateMetrics(y_test, vote_preds, vote_sum/4);
    
    % Mostrar matriz de confusión
    conf_mat_vote = confusionmat(y_test, vote_preds);
    fprintf('Matriz de confusión Ensamble por Votación:\n');
    disp(conf_mat_vote);
    
    % Mostrar métricas
    fprintf('Métricas Ensamble por Votación:\n');
    fprintf('  - Exactitud: %.4f\n', acc_vote);
    fprintf('  - Precisión: %.4f\n', prec_vote);
    fprintf('  - Sensibilidad: %.4f\n', sens_vote);
    fprintf('  - F1-Score: %.4f\n', f1_vote);
    
catch ens_error
    fprintf('ERROR: Problema con el modelo de Ensamble Avanzado: %s\n', ens_error.message);
end

%% 12. VALIDACIÓN CRUZADA DEL MEJOR MODELO
fprintf('\n12. VALIDACIÓN CRUZADA DEL MEJOR MODELO\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    % Recopilar métricas de todos los modelos optimizados
    model_metrics = struct();
    
    % Añadir métricas de cada modelo optimizado
    if exist('f1_rf_opt', 'var') && exist('auc_rf', 'var')
        model_metrics.rf = struct('name', 'Random Forest', 'f1', f1_rf_opt, 'auc', auc_rf, 'thresh', best_thresh_rf);
    end
    
    if exist('f1_gb_opt', 'var') && exist('auc_gb', 'var')
        model_metrics.gb = struct('name', 'Gradient Boosting', 'f1', f1_gb_opt, 'auc', auc_gb, 'thresh', best_thresh_gb);
    end
    
    if exist('f1_rus_opt', 'var') && exist('auc_rus', 'var')
        model_metrics.rus = struct('name', 'RUSBoost', 'f1', f1_rus_opt, 'auc', auc_rus, 'thresh', best_thresh_rus);
    end
    
    if exist('f1_nn_opt', 'var') && exist('auc_nn', 'var')
        model_metrics.nn = struct('name', 'Red Neuronal', 'f1', f1_nn_opt, 'auc', auc_nn, 'thresh', best_thresh_nn);
    end
    
    if exist('f1_ens_opt', 'var') && exist('auc_ens', 'var')
        model_metrics.ens = struct('name', 'Ensamble Avanzado', 'f1', f1_ens_opt, 'auc', auc_ens, 'thresh', best_thresh_ens);
    end
    
    if exist('f1_vote', 'var')
        model_metrics.vote = struct('name', 'Ensamble por Votación', 'f1', f1_vote, 'auc', NaN, 'thresh', 0.5);
    end
    
    % Identificar el mejor modelo según F1-Score y AUC-ROC
    model_names = fieldnames(model_metrics);
    f1_scores = zeros(length(model_names), 1);
    auc_values = zeros(length(model_names), 1);
    
    for i = 1:length(model_names)
        model_name = model_names{i};
        f1_scores(i) = model_metrics.(model_name).f1;
        
        if isfield(model_metrics.(model_name), 'auc') && ~isnan(model_metrics.(model_name).auc)
            auc_values(i) = model_metrics.(model_name).auc;
        else
            auc_values(i) = 0;
        end
    end
    
    % Mejor modelo según F1-Score
    [best_f1, best_f1_idx] = max(f1_scores);
    best_f1_model = model_metrics.(model_names{best_f1_idx}).name;
    
    % Mejor modelo según AUC-ROC
    [best_auc, best_auc_idx] = max(auc_values);
    best_auc_model = model_metrics.(model_names{best_auc_idx}).name;
    
    fprintf('Mejor modelo según F1-Score: %s (F1=%.4f, AUC=%.4f)\n', ...
        best_f1_model, best_f1, auc_values(best_f1_idx));
    fprintf('Mejor modelo según AUC-ROC: %s (AUC=%.4f, F1=%.4f)\n', ...
        best_auc_model, best_auc, f1_scores(best_auc_idx));
    
    % Seleccionar el mejor modelo para validación cruzada (el que tiene mejor F1-Score)
    best_model_type = model_names{best_f1_idx};
    best_model_name = model_metrics.(best_model_type).name;
    best_model_thresh = model_metrics.(best_model_type).thresh;
    
    fprintf('\nRealizando validación cruzada completa para el mejor modelo: %s\n', best_model_name);
    
    % Definir número de folds y repeticiones para validación cruzada
    k_folds = VALIDATION_FOLDS;
    n_repeats = VALIDATION_REPEATS;
    
    % Matrices para almacenar resultados
    cv_metrics = zeros(n_repeats, k_folds, 5); % Accuracy, Precision, Recall, F1, AUC
    
    % Preparar datos completos
    X_full = X_all;
    y_full = y_all;
    
    % Para modelos que requieren solo variables numéricas
    numeric_vars_idx_full = varfun(@isnumeric, X_full, 'OutputFormat', 'uniform');
    X_numeric_full = X_full(:, numeric_vars_idx_full);
    X_numeric_array = table2array(X_numeric_full);
    
    % Aplicar normalización si es necesario
    if strcmp(best_model_type, 'nn')
        % Normalizar datos
        [X_numeric_norm, mu, sigma] = normalize(X_numeric_array, 1);
        X_numeric_array = X_numeric_norm;
        
        % Manejar posibles NaN o Inf
        X_numeric_array(isnan(X_numeric_array)) = 0;
        X_numeric_array(isinf(X_numeric_array)) = 0;
    end
    
    for rep = 1:n_repeats
        fprintf('  Repetición %d de %d...\n', rep, n_repeats);
        
        % Crear partición para esta repetición
        cv = cvpartition(y_full, 'KFold', k_folds, 'Stratify', true);
        
        for fold = 1:k_folds
            fprintf('    Fold %d de %d...\n', fold, k_folds);
            
            % Obtener índices de entrenamiento y prueba
            train_idx = cv.training(fold);
            test_idx = cv.test(fold);
            
            % Preparar datos para este fold
            X_train_fold = X_full(train_idx, :);
            y_train_fold = y_full(train_idx);
            X_test_fold = X_full(test_idx, :);
            y_test_fold = y_full(test_idx);
            
            % Para modelos que requieren solo variables numéricas
            X_train_num_fold = X_numeric_array(train_idx, :);
            X_test_num_fold = X_numeric_array(test_idx, :);
            
            % Entrenar modelo según el tipo seleccionado
            try
                switch best_model_type
                    case 'rf'
                        % Random Forest
                        model_fold = TreeBagger(best_rf_params.NumTrees, X_train_num_fold, y_train_fold, ...
                                              'Method', 'classification', 'MinLeafSize', best_rf_params.MinLeafSize);
                        [~, scores_fold] = predict(model_fold, X_test_num_fold);
                        scores_to_use = scores_fold(:,2);
                        
                    case 'gb'
                        % Gradient Boosting
                        template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
                        model_fold = fitcensemble(X_train_num_fold, y_train_fold, 'Method', 'GentleBoost', ...
                                               'NumLearningCycles', best_gb_params.NumLearnCycles, ...
                                               'Learners', template, 'LearnRate', best_gb_params.LearnRate);
                        [~, scores_fold] = predict(model_fold, X_test_num_fold);
                        scores_to_use = scores_fold(:,2);
                        
                    case 'rus'
                        % RUSBoost
                        template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
                        model_fold = fitcensemble(X_train_num_fold, y_train_fold, 'Method', 'RUSBoost', ...
                                               'NumLearningCycles', 100, 'Learners', template, ...
                                               'LearnRate', best_rus_params.LearnRate, ...
                                               'RatioToSmallest', best_rus_params.RatioToSmallest);
                        [~, scores_fold] = predict(model_fold, X_test_num_fold);
                        scores_to_use = scores_fold(:,2);
                        
                    case 'nn'
                        % Red Neuronal
                        net_fold = patternnet(best_nn_params.HiddenLayers);
                        net_fold.trainParam.showWindow = false;
                        net_fold.trainParam.epochs = 100;
                        net_fold.divideParam.trainRatio = 0.7;
                        net_fold.divideParam.valRatio = 0.15;
                        net_fold.divideParam.testRatio = 0.15;
                        
                        [net_fold, ~] = train(net_fold, X_train_num_fold', y_train_fold');
                        
                        % Realizar predicciones
                        scores_fold = net_fold(X_test_num_fold')';
                        scores_to_use = scores_fold;
                        
                    case 'ens'
                        % Para ensamble, necesitamos entrenar todos los modelos individuales
                        % Entrenar RF
                        rf_model_fold = TreeBagger(best_rf_params.NumTrees, X_train_num_fold, y_train_fold, ...
                                                'Method', 'classification', 'MinLeafSize', best_rf_params.MinLeafSize);
                        [~, rf_scores_fold] = predict(rf_model_fold, X_test_num_fold);
                        
                        % Entrenar GB
                        template = templateTree('MaxNumSplits', 20, 'MinLeafSize', 5);
                        gb_model_fold = fitcensemble(X_train_num_fold, y_train_fold, 'Method', 'GentleBoost', ...
                                                  'NumLearningCycles', best_gb_params.NumLearnCycles, ...
                                                  'Learners', template, 'LearnRate', best_gb_params.LearnRate);
                        [~, gb_scores_fold] = predict(gb_model_fold, X_test_num_fold);
                        
                        % Entrenar RUSBoost
                        rus_model_fold = fitcensemble(X_train_num_fold, y_train_fold, 'Method', 'RUSBoost', ...
                                                   'NumLearningCycles', 100, 'Learners', template, ...
                                                   'LearnRate', best_rus_params.LearnRate, ...
                                                   'RatioToSmallest', best_rus_params.RatioToSmallest);
                        [~, rus_scores_fold] = predict(rus_model_fold, X_test_num_fold);
                        
                        % Entrenar Red Neuronal
                        net_fold = patternnet(best_nn_params.HiddenLayers);
                        net_fold.trainParam.showWindow = false;
                        net_fold.trainParam.epochs = 100;
                        net_fold.divideParam.trainRatio = 0.7;
                        net_fold.divideParam.valRatio = 0.15;
                        net_fold.divideParam.testRatio = 0.15;
                        
                        [net_fold, ~] = train(net_fold, X_train_num_fold', y_train_fold');
                        nn_scores_fold = net_fold(X_test_num_fold')';
                        
                        % Calibrar probabilidades
                        % (Simplificado para validación cruzada)
                        
                        % Combinar scores con pesos
                        all_scores_fold = [rf_scores_fold(:,2), gb_scores_fold(:,2), rus_scores_fold(:,2), nn_scores_fold];
                        scores_to_use = all_scores_fold * weights';
                        
                    otherwise
                        error('Tipo de modelo no reconocido: %s', best_model_type);
                end
                
                % Aplicar umbral óptimo para obtener predicciones finales
                preds_fold = double(scores_to_use >= best_model_thresh);
                
                % Calcular métricas
                [acc, prec, sens, f1, auc] = calculateMetrics(y_test_fold, preds_fold, scores_to_use);
                cv_metrics(rep, fold, :) = [acc, prec, sens, f1, auc];
                
            catch fold_error
                fprintf('      Error en este fold: %s\n', fold_error.message);
                % Asignar valores NaN para este fold
                cv_metrics(rep, fold, :) = NaN(1, 5);
            end
        end
    end
    
    % Calcular promedios y desviaciones estándar
    cv_means = nanmean(reshape(cv_metrics, [], 5), 1);
    cv_stds = nanstd(reshape(cv_metrics, [], 5), 0, 1);
    
    % Mostrar resultados
    fprintf('\nResultados de validación cruzada para %s (%dx%d-fold):\n', best_model_name, n_repeats, k_folds);
    fprintf('  - Exactitud: %.4f (±%.4f)\n', cv_means(1), cv_stds(1));
    fprintf('  - Precisión: %.4f (±%.4f)\n', cv_means(2), cv_stds(2));
    fprintf('  - Sensibilidad: %.4f (±%.4f)\n', cv_means(3), cv_stds(3));
    fprintf('  - F1-Score: %.4f (±%.4f)\n', cv_means(4), cv_stds(4));
    fprintf('  - AUC-ROC: %.4f (±%.4f)\n', cv_means(5), cv_stds(5));
    
    % Crear gráfico de caja para mostrar distribución de métricas si está habilitada la opción
    if SAVE_FIGURES
        % Reorganizar datos para boxplot
        cv_data_reshaped = reshape(cv_metrics, [], 5);
        
        % Eliminar valores NaN
        valid_idx = ~any(isnan(cv_data_reshaped), 2);
        cv_data_valid = cv_data_reshaped(valid_idx, :);
        
        if ~isempty(cv_data_valid)
            figure;
            boxplot(cv_data_valid, 'Labels', {'Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score', 'AUC-ROC'});
            title(sprintf('Distribución de métricas en validación cruzada - %s', best_model_name));
            grid on;
            saveas(gcf, 'cv_metrics_boxplot.png');
            fprintf('Gráfico de distribución de métricas guardado en "cv_metrics_boxplot.png"\n');
        end
    end
catch cv_error
    fprintf('ERROR: Problema en validación cruzada: %s\n', cv_error.message);
end
%% 13. COMPARACIÓN FINAL DE TODOS LOS MODELOS
fprintf('\n13. COMPARACIÓN FINAL DE TODOS LOS MODELOS\n');
fprintf('---------------------------------------------------------------------------------\n');

try
    % Recopilar resultados de todos los modelos en tablas para mejor comparación
    fprintf('Comparación de todos los modelos (con umbrales optimizados):\n');
    fprintf('------------------------------------------------------------------------------------------------------------\n');
    fprintf('Modelo                 | Exactitud | Precisión | Sensibilidad | F1-Score | AUC-ROC | Umbral\n');
    fprintf('------------------------------------------------------------------------------------------------------------\n');
    
    % Random Forest
    if exist('acc_rf_opt', 'var')
        fprintf('Random Forest          | %.4f    | %.4f    | %.4f       | %.4f   | %.4f  | %.2f\n', ...
            acc_rf_opt, prec_rf_opt, sens_rf_opt, f1_rf_opt, auc_rf, best_thresh_rf);
    end
    
    % Gradient Boosting
    if exist('acc_gb_opt', 'var')
        fprintf('Gradient Boosting      | %.4f    | %.4f    | %.4f       | %.4f   | %.4f  | %.2f\n', ...
            acc_gb_opt, prec_gb_opt, sens_gb_opt, f1_gb_opt, auc_gb, best_thresh_gb);
    end
    
    % RUSBoost
    if exist('acc_rus_opt', 'var')
        fprintf('RUSBoost               | %.4f    | %.4f    | %.4f       | %.4f   | %.4f  | %.2f\n', ...
            acc_rus_opt, prec_rus_opt, sens_rus_opt, f1_rus_opt, auc_rus, best_thresh_rus);
    end
    
    % Red Neuronal
    if exist('acc_nn_opt', 'var')
        fprintf('Red Neuronal           | %.4f    | %.4f    | %.4f       | %.4f   | %.4f  | %.2f\n', ...
            acc_nn_opt, prec_nn_opt, sens_nn_opt, f1_nn_opt, auc_nn, best_thresh_nn);
    end
    
    % Ensamble Avanzado
    if exist('acc_ens_opt', 'var')
        fprintf('Ensamble Avanzado      | %.4f    | %.4f    | %.4f       | %.4f   | %.4f  | %.2f\n', ...
            acc_ens_opt, prec_ens_opt, sens_ens_opt, f1_ens_opt, auc_ens, best_thresh_ens);
    end
    
    % Ensamble por Votación
    if exist('acc_vote', 'var')
        fprintf('Ensamble por Votación  | %.4f    | %.4f    | %.4f       | %.4f   | N/A     | N/A\n', ...
            acc_vote, prec_vote, sens_vote, f1_vote);
    end
    
    fprintf('------------------------------------------------------------------------------------------------------------\n');
    
    % Visualizar comparación de curvas ROC si está habilitada la opción
    if SAVE_FIGURES
        figure;
        hold on;
        
        % Colores para cada modelo
        colors = {'b', 'r', 'm', 'g', 'c', 'k'};
        legend_entries = {};
        plot_count = 0;
        
        % Random Forest
        if exist('scores_rf', 'var')
            plot_count = plot_count + 1;
            [x, y, ~, ~] = perfcurve(y_test, scores_rf(:,2), '1');
            plot(x, y, [colors{plot_count} '-'], 'LineWidth', 2);
            legend_entries{end+1} = sprintf('Random Forest (AUC=%.4f)', auc_rf);
        end
        
        % Gradient Boosting
        if exist('scores_gb', 'var')
            plot_count = plot_count + 1;
            [x, y, ~, ~] = perfcurve(y_test, scores_gb(:,2), '1');
            plot(x, y, [colors{plot_count} '-'], 'LineWidth', 2);
            legend_entries{end+1} = sprintf('Gradient Boosting (AUC=%.4f)', auc_gb);
        end
        
        % RUSBoost
        if exist('rusboost_scores', 'var')
            plot_count = plot_count + 1;
            [x, y, ~, ~] = perfcurve(y_test, rusboost_scores(:,2), '1');
            plot(x, y, [colors{plot_count} '-'], 'LineWidth', 2);
            legend_entries{end+1} = sprintf('RUSBoost (AUC=%.4f)', auc_rus);
        end
        
        % Red Neuronal
        if exist('nn_scores', 'var')
            plot_count = plot_count + 1;
            [x, y, ~, ~] = perfcurve(y_test, nn_scores, '1');
            plot(x, y, [colors{plot_count} '-'], 'LineWidth', 2);
            legend_entries{end+1} = sprintf('Red Neuronal (AUC=%.4f)', auc_nn);
        end
        
        % Ensamble Avanzado
        if exist('weighted_scores', 'var')
            plot_count = plot_count + 1;
            [x, y, ~, ~] = perfcurve(y_test, weighted_scores, '1');
            plot(x, y, [colors{plot_count} '-'], 'LineWidth', 2);
            legend_entries{end+1} = sprintf('Ensamble Avanzado (AUC=%.4f)', auc_ens);
        end
        
        % Añadir línea de referencia
        plot([0, 1], [0, 1], 'k--');
        
        % Configurar gráfico
        xlabel('Tasa de Falsos Positivos', 'FontSize', 12);
        ylabel('Tasa de Verdaderos Positivos', 'FontSize', 12);
        title('Comparación de Curvas ROC - Todos los modelos', 'FontSize', 14);
        legend(legend_entries, 'Location', 'southeast');
        grid on;
        
        % Guardar gráfico
        saveas(gcf, 'all_models_roc_comparison.png');
        fprintf('Comparación de curvas ROC guardada en "all_models_roc_comparison.png"\n');
        
        % Comparación de curvas de calibración
        if exist('rf_scores_cal', 'var') && exist('gb_scores_cal', 'var') && ...
           exist('rus_scores_cal', 'var') && exist('nn_scores_cal', 'var')
            
            figure;
            hold on;
            
            % Crear línea diagonal de referencia
            plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);
            
            % Crear bins para las probabilidades predichas
            n_bins = 10;
            bin_edges = linspace(0, 1, n_bins + 1);
            bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
            
            % Random Forest
            bin_indices_rf = discretize(rf_scores_cal, bin_edges);
            observed_rf = zeros(1, n_bins);
            for i = 1:n_bins
                bin_samples = y_test(bin_indices_rf == i);
                if ~isempty(bin_samples)
                    observed_rf(i) = mean(bin_samples);
                else
                    observed_rf(i) = NaN;
                end
            end
            plot(bin_centers, observed_rf, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6);
            
            % Gradient Boosting
            bin_indices_gb = discretize(gb_scores_cal, bin_edges);
            observed_gb = zeros(1, n_bins);
            for i = 1:n_bins
                bin_samples = y_test(bin_indices_gb == i);
                if ~isempty(bin_samples)
                    observed_gb(i) = mean(bin_samples);
                else
                    observed_gb(i) = NaN;
                end
            end
            plot(bin_centers, observed_gb, 'ro-', 'LineWidth', 1.5, 'MarkerSize', 6);
            
            % RUSBoost
            bin_indices_rus = discretize(rus_scores_cal, bin_edges);
            observed_rus = zeros(1, n_bins);
            for i = 1:n_bins
                bin_samples = y_test(bin_indices_rus == i);
                if ~isempty(bin_samples)
                    observed_rus(i) = mean(bin_samples);
                else
                    observed_rus(i) = NaN;
                end
            end
            plot(bin_centers, observed_rus, 'mo-', 'LineWidth', 1.5, 'MarkerSize', 6);
            
            % Red Neuronal
            bin_indices_nn = discretize(nn_scores_cal, bin_edges);
            observed_nn = zeros(1, n_bins);
            for i = 1:n_bins
                bin_samples = y_test(bin_indices_nn == i);
                if ~isempty(bin_samples)
                    observed_nn(i) = mean(bin_samples);
                else
                    observed_nn(i) = NaN;
                end
            end
            plot(bin_centers, observed_nn, 'go-', 'LineWidth', 1.5, 'MarkerSize', 6);
            
            % Ensamble Avanzado
            bin_indices_ens = discretize(weighted_scores, bin_edges);
            observed_ens = zeros(1, n_bins);
            for i = 1:n_bins
                bin_samples = y_test(bin_indices_ens == i);
                if ~isempty(bin_samples)
                    observed_ens(i) = mean(bin_samples);
                else
                    observed_ens(i) = NaN;
                end
            end
            plot(bin_centers, observed_ens, 'co-', 'LineWidth', 1.5, 'MarkerSize', 6);
            
            % Configurar gráfico
            xlabel('Probabilidad predicha', 'FontSize', 12);
            ylabel('Proporción observada', 'FontSize', 12);
            title('Comparación de Calibración - Todos los modelos', 'FontSize', 14);
            legend({'Referencia perfecta', 'Random Forest', 'Gradient Boosting', 'RUSBoost', 'Red Neuronal', 'Ensamble Avanzado'}, 'Location', 'southeast');
            grid on;
            axis square;
            
            % Guardar gráfico
            saveas(gcf, 'all_models_calibration_comparison.png');
            fprintf('Comparación de calibración guardada en "all_models_calibration_comparison.png"\n');
        end
    end
catch comp_error
    fprintf('ERROR: Problema en comparación final: %s\n', comp_error.message);
end
%% 14. EXPORTAR EL MEJOR MODELO PARA APLICACIÓN WEB
fprintf('\n14. EXPORTAR EL MEJOR MODELO PARA APLICACIÓN WEB\n');
fprintf('---------------------------------------------------------------------------------\n');

if EXPORT_MODEL
    try
        fprintf('Exportando el mejor modelo para uso en aplicación web...\n');
        
        % Identificar el mejor modelo para aplicaciones clínicas (basado en F1-Score)
        % El F1-Score es mejor métrica cuando se busca equilibrio entre precisión y sensibilidad
        best_model_name = best_f1_model;
        best_model_type = model_names{best_f1_idx};
        best_model_thresh = model_metrics.(best_model_type).thresh;
        
        fprintf('Modelo seleccionado para exportación: %s\n', best_model_name);
        fprintf('Umbral óptimo: %.4f\n', best_model_thresh);
        fprintf('Métricas de rendimiento:\n');
        fprintf('  - F1-Score: %.4f\n', best_f1);
        fprintf('  - AUC-ROC: %.4f\n', auc_values(best_f1_idx));
        
        % Crear estructura para exportación
        export_model = struct();
        export_model.model_name = best_model_name;
        export_model.creation_date = datestr(now);
        export_model.threshold = best_model_thresh;
        
        % Metadatos del modelo
        export_model.metadata = struct();
        export_model.metadata.description = 'Modelo avanzado de predicción de diabetes en pacientes UCI (WiDS Datathon 2021)';
        export_model.metadata.creator = 'Engineering for the Americas Virtual Hackathon 2025';
        export_model.metadata.version = '2.0';
        export_model.metadata.date = datestr(now);
        export_model.metadata.f1_score = best_f1;
        export_model.metadata.auc_roc = auc_values(best_f1_idx);
        
        % Añadir resultados de validación cruzada
        if exist('cv_means', 'var') && exist('cv_stds', 'var')
            export_model.metadata.cv_results = struct();
            export_model.metadata.cv_results.accuracy = struct('mean', cv_means(1), 'std', cv_stds(1));
            export_model.metadata.cv_results.precision = struct('mean', cv_means(2), 'std', cv_stds(2));
            export_model.metadata.cv_results.sensitivity = struct('mean', cv_means(3), 'std', cv_stds(3));
            export_model.metadata.cv_results.f1_score = struct('mean', cv_means(4), 'std', cv_stds(4));
            export_model.metadata.cv_results.auc_roc = struct('mean', cv_means(5), 'std', cv_stds(5));
        end
        
        % Guardar características utilizadas
        export_model.features = struct();
        export_model.features.selected_features = X_train.Properties.VariableNames;
        export_model.features.num_features = width(X_train);
        
        % Top características importantes (si aplica)
        if strcmp(best_model_type, 'rf') && exist('imp', 'var') && exist('idx', 'var')
            export_model.features.top_important = cell(min(10, length(idx)), 2);
            for i = 1:min(10, length(idx))
                export_model.features.top_important{i, 1} = var_names{idx(i)};
                export_model.features.top_important{i, 2} = sorted_imp(i);
            end
        end
        
        % Guardar información para preprocesamiento
        export_model.preprocessing = struct();
        
        % Estadísticas para imputación de valores faltantes
        export_model.preprocessing.imputation = struct();
        
        % Características clave para la predicción (basado en correlaciones y análisis)
        key_features = {'d1_glucose_max', 'bmi', 'age', 'd1_creatinine_max', 'd1_bun_max', 'glucose_apache'};
        
        for i = 1:length(key_features)
            feature = key_features{i};
            if ~ismember(feature, X_train.Properties.VariableNames)
                continue;
            end
            
            export_model.preprocessing.imputation.(feature) = struct();
            
            % Estadísticas globales
            export_model.preprocessing.imputation.(feature).mean = mean(X_train.(feature), 'omitnan');
            export_model.preprocessing.imputation.(feature).median = median(X_train.(feature), 'omitnan');
            
            % Estadísticas por clase
            diabetes_values = X_train.(feature)(y_train == 1);
            no_diabetes_values = X_train.(feature)(y_train == 0);
            
            export_model.preprocessing.imputation.(feature).diabetes_mean = mean(diabetes_values, 'omitnan');
            export_model.preprocessing.imputation.(feature).diabetes_median = median(diabetes_values, 'omitnan');
            export_model.preprocessing.imputation.(feature).no_diabetes_mean = mean(no_diabetes_values, 'omitnan');
            export_model.preprocessing.imputation.(feature).no_diabetes_median = median(no_diabetes_values, 'omitnan');
        end
        
        % Si es un modelo normalizado, guardar parámetros de normalización
        if strcmp(best_model_type, 'nn')
            export_model.preprocessing.normalization = struct('mean', norm_params.mean, 'std', norm_params.std);
        end
        
        % Guardar el modelo según su tipo
        switch best_model_type
            case 'rf'
                % Random Forest
                export_model.algorithm = rf_model;
                export_model.predict_function = @predictDiabetesRF;
                
            case 'gb'
                % Gradient Boosting
                export_model.algorithm = gb_model;
                export_model.predict_function = @predictDiabetesGB;
                
            case 'rus'
                % RUSBoost
                export_model.algorithm = rusboost_model;
                export_model.predict_function = @predictDiabetesRUSBoost;
                
            case 'nn'
                % Red Neuronal
                export_model.algorithm = net;
                export_model.predict_function = @predictDiabetesNN;
                
            case 'ens'
                % Ensamble Avanzado
                export_model.algorithm = struct();
                export_model.algorithm.rf_model = rf_model;
                export_model.algorithm.gb_model = gb_model;
                export_model.algorithm.rusboost_model = rusboost_model;
                export_model.algorithm.nn_model = net;
                
                % Guardar pesos para el ensamble
                export_model.algorithm.weights = weights;
                export_model.algorithm.models = {'Random Forest', 'Gradient Boosting', 'RUSBoost', 'Red Neuronal'};
                export_model.algorithm.combine_method = 'weighted_average';
                export_model.predict_function = @predictDiabetesEnsemble;
                
            case 'vote'
                % Ensamble por Votación
                export_model.algorithm = struct();
                export_model.algorithm.rf_model = rf_model;
                export_model.algorithm.gb_model = gb_model;
                export_model.algorithm.rusboost_model = rusboost_model;
                export_model.algorithm.nn_model = net;
                export_model.algorithm.combine_method = 'voting';
                export_model.predict_function = @predictDiabetesVoting;
        end
        
        % Información para API web
        export_model.web_api = struct();
        export_model.web_api.version = '2.0';
        export_model.web_api.input_format = 'JSON';
        export_model.web_api.output_format = 'JSON';
        export_model.web_api.required_fields = key_features;
        export_model.web_api.endpoint = '/api/diabetes-prediction';
        
        % Ejemplo de uso del modelo
        export_model.web_api.example_input = '{"d1_glucose_max": 180, "bmi": 32.5, "age": 65, "d1_creatinine_max": 1.2}';
        export_model.web_api.example_output = '{"prediction": 1, "probability": 0.87, "risk_level": "high"}';
        
        % Guardar modelo en un archivo .mat
        model_filename = ['diabetes_model_web_api_', datestr(now, 'yyyymmdd'), '.mat'];
        save(model_filename, 'export_model');
        fprintf('Modelo exportado exitosamente como "%s"\n', model_filename);
        
    catch export_error
        fprintf('ERROR: Problema al exportar el modelo: %s\n', export_error.message);
    end
else
    fprintf('Exportación de modelo desactivada. Cambie EXPORT_MODEL=true para habilitar.\n');
end

%% 15. RESUMEN Y CONCLUSIONES
fprintf('\n15. RESUMEN Y CONCLUSIONES\n');
fprintf('---------------------------------------------------------------------------------\n');

fprintf('Resumen del estudio de predicción de diabetes:\n\n');

% 1. Resumen de datos
fprintf('1. DATOS:\n');
fprintf('   - Conjunto de datos: WiDS Datathon 2021\n');
fprintf('   - Dimensiones originales: %d filas, %d columnas\n', height(training_data), width(training_data));
fprintf('   - Distribución de clases: %.1f%% con diabetes, %.1f%% sin diabetes\n', ...
    100 * mean(training_data.(TARGET_VAR)), 100 * (1 - mean(training_data.(TARGET_VAR))));

% 2. Resumen de preprocesamiento
fprintf('\n2. PREPROCESAMIENTO:\n');
fprintf('   - Variables con valores faltantes tratadas mediante imputación estratificada\n');
fprintf('   - Outliers limitados utilizando método IQR con umbral de %.1f\n', OUTLIER_THRESHOLD);
fprintf('   - Variables con >%.0f%% de valores faltantes eliminadas\n', MISSING_THRESHOLD*100);

% 3. Resumen de ingeniería de características
fprintf('\n3. INGENIERÍA DE CARACTERÍSTICAS:\n');
fprintf('   - Se crearon características médicamente relevantes (ej. rango de glucosa, categorías de BMI)\n');
fprintf('   - Se generaron interacciones entre variables clave (ej. glucosa-edad, glucosa-BMI)\n');
fprintf('   - Se incluyeron variables indicadoras de comorbilidades y estado crítico\n');
fprintf('   - Se seleccionaron %d características finales mediante correlación y Random Forest\n', ...
    width(X_selected));

% 4. Resumen de modelos
fprintf('\n4. MODELOS EVALUADOS:\n');
fprintf('   - Random Forest (RF): %.4f F1-Score, %.4f AUC-ROC\n', f1_rf_opt, auc_rf);
fprintf('   - Gradient Boosting (GB): %.4f F1-Score, %.4f AUC-ROC\n', f1_gb_opt, auc_gb);
fprintf('   - RUSBoost (para datos desbalanceados): %.4f F1-Score, %.4f AUC-ROC\n', f1_rus_opt, auc_rus);
fprintf('   - Red Neuronal (NN): %.4f F1-Score, %.4f AUC-ROC\n', f1_nn_opt, auc_nn);
fprintf('   - Ensamble Avanzado: %.4f F1-Score, %.4f AUC-ROC\n', f1_ens_opt, auc_ens);
fprintf('   - Ensamble por Votación: %.4f F1-Score\n', f1_vote);

% 5. Mejor modelo
fprintf('\n5. MEJOR MODELO: %s\n', best_model_name);
fprintf('   - F1-Score: %.4f\n', best_f1);
fprintf('   - AUC-ROC: %.4f\n', auc_values(best_f1_idx));
fprintf('   - Umbral óptimo: %.4f\n', best_model_thresh);
fprintf('   - Exactitud CV: %.4f (±%.4f)\n', cv_means(1), cv_stds(1));
fprintf('   - Precisión CV: %.4f (±%.4f)\n', cv_means(2), cv_stds(2));
fprintf('   - Sensibilidad CV: %.4f (±%.4f)\n', cv_means(3), cv_stds(3));
%
% 6. Variables más importantes
if strcmp(best_model_type, 'rf') && exist('imp', 'var') && exist('idx', 'var')
    fprintf('\n6. VARIABLES MÁS IMPORTANTES (según %s):\n', best_model_name);
    for i = 1:min(5, length(idx))
        fprintf('   - %s: %.4f\n', var_names{idx(i)}, sorted_imp(i));
    end
end

% 7. Conclusiones
fprintf('\n7. CONCLUSIONES:\n');
fprintf('   - Los niveles de glucosa son el predictor más fuerte para diabetes\n');
fprintf('   - El BMI y la edad también son factores importantes\n');
fprintf('   - Los ensambles mejoran el rendimiento sobre modelos individuales\n');
fprintf('   - La calibración de probabilidades es crucial para interpretabilidad clínica\n');
fprintf('   - El modelo exportado puede utilizarse en una aplicación web para asistencia clínica\n\n');

fprintf('===================================================================================\n');
fprintf('                PROCESO DE MODELADO COMPLETADO EXITOSAMENTE                        \n');
fprintf('===================================================================================\n\n');

%% Funciones para predicción con el modelo exportado

% Función para Random Forest
function [probabilities, predictions] = predictDiabetesRF(model, new_data)
    try
        % Verificar si los datos tienen las variables necesarias
        check_missing_fields(model, new_data);
        
        % Realizar predicciones
        [~, scores] = predict(model.algorithm, new_data);
        probabilities = scores(:,2);
        
        % Aplicar umbral para obtener predicciones binarias
        predictions = double(probabilities >= model.threshold);
    catch predict_error
        error('Error al realizar predicciones con Random Forest: %s', predict_error.message);
    end
end

% Función para Gradient Boosting
function [probabilities, predictions] = predictDiabetesGB(model, new_data)
    try
        % Verificar si los datos tienen las variables necesarias
        check_missing_fields(model, new_data);
        
        % Obtener solo variables numéricas
        numeric_vars_idx = varfun(@isnumeric, new_data, 'OutputFormat', 'uniform');
        X_numeric = new_data(:, numeric_vars_idx);
        X_matrix = table2array(X_numeric);
        
        % Predecir
        [~, scores] = predict(model.algorithm, X_matrix);
        probabilities = scores(:,2);
        
        % Aplicar umbral para obtener predicciones binarias
        predictions = double(probabilities >= model.threshold);
    catch predict_error
        error('Error al realizar predicciones con Gradient Boosting: %s', predict_error.message);
    end
end

% Función para RUSBoost
function [probabilities, predictions] = predictDiabetesRUSBoost(model, new_data)
    try
        % Verificar si los datos tienen las variables necesarias
        check_missing_fields(model, new_data);
        
        % Obtener solo variables numéricas
        numeric_vars_idx = varfun(@isnumeric, new_data, 'OutputFormat', 'uniform');
        X_numeric = new_data(:, numeric_vars_idx);
        X_matrix = table2array(X_numeric);
        
        % Predecir
        [~, scores] = predict(model.algorithm, X_matrix);
        probabilities = scores(:,2);
        
        % Aplicar umbral para obtener predicciones binarias
        predictions = double(probabilities >= model.threshold);
    catch predict_error
        error('Error al realizar predicciones con RUSBoost: %s', predict_error.message);
    end
end

% Función para Red Neuronal
function [probabilities, predictions] = predictDiabetesNN(model, new_data)
    try
        % Verificar si los datos tienen las variables necesarias
        check_missing_fields(model, new_data);
        
        % Obtener solo variables numéricas
        numeric_vars_idx = varfun(@isnumeric, new_data, 'OutputFormat', 'uniform');
        X_numeric = new_data(:, numeric_vars_idx);
        X_matrix = table2array(X_numeric);
        
        % Normalizar
        if isfield(model.preprocessing, 'normalization')
            X_norm = (X_matrix - model.preprocessing.normalization.mean) ./ model.preprocessing.normalization.std;
            X_norm(isnan(X_norm)) = 0;
            X_norm(isinf(X_norm)) = 0;
        else
            X_norm = X_matrix;
        end
        
        % Predecir
        output = model.algorithm(X_norm');
        probabilities = output';
        
        % Aplicar umbral para obtener predicciones binarias
        predictions = double(probabilities >= model.threshold);
    catch predict_error
        error('Error al realizar predicciones con Red Neuronal: %s', predict_error.message);
    end
end

% Función para Ensamble Avanzado
function [probabilities, predictions] = predictDiabetesEnsemble(model, new_data)
    try
        % Verificar si los datos tienen las variables necesarias
        check_missing_fields(model, new_data);
        
        % Obtener solo variables numéricas
        numeric_vars_idx = varfun(@isnumeric, new_data, 'OutputFormat', 'uniform');
        X_numeric = new_data(:, numeric_vars_idx);
        X_matrix = table2array(X_numeric);
        
        % Preparar matriz normalizada para Red Neuronal
        if isfield(model.preprocessing, 'normalization')
            X_norm = (X_matrix - model.preprocessing.normalization.mean) ./ model.preprocessing.normalization.std;
            X_norm(isnan(X_norm)) = 0;
            X_norm(isinf(X_norm)) = 0;
        else
            X_norm = X_matrix;
        end
        
        % Obtener predicciones de los modelos individuales
        [~, scores_rf] = predict(model.algorithm.rf_model, new_data);
        [~, scores_gb] = predict(model.algorithm.gb_model, X_matrix);
        [~, scores_rus] = predict(model.algorithm.rusboost_model, X_matrix);
        nn_scores = model.algorithm.nn_model(X_norm')';
        
        % Combinar predicciones según los pesos
        all_scores = [scores_rf(:,2), scores_gb(:,2), scores_rus(:,2), nn_scores];
        probabilities = all_scores * model.algorithm.weights';
        
        % Aplicar umbral para obtener predicciones binarias
        predictions = double(probabilities >= model.threshold);
    catch predict_error
        error('Error al realizar predicciones con Ensamble Avanzado: %s', predict_error.message);
    end
end

% Función para Ensamble por Votación
function [probabilities, predictions] = predictDiabetesVoting(model, new_data)
    try
        % Verificar si los datos tienen las variables necesarias
        check_missing_fields(model, new_data);
        
        % Obtener solo variables numéricas
        numeric_vars_idx = varfun(@isnumeric, new_data, 'OutputFormat', 'uniform');
        X_numeric = new_data(:, numeric_vars_idx);
        X_matrix = table2array(X_numeric);
        
        % Preparar matriz normalizada para Red Neuronal
        if isfield(model.preprocessing, 'normalization')
            X_norm = (X_matrix - model.preprocessing.normalization.mean) ./ model.preprocessing.normalization.std;
            X_norm(isnan(X_norm)) = 0;
            X_norm(isinf(X_norm)) = 0;
        else
            X_norm = X_matrix;
        end
        
        % Obtener predicciones binarias de los modelos individuales
        [~, scores_rf] = predict(model.algorithm.rf_model, new_data);
        pred_rf = double(scores_rf(:,2) >= 0.5);
        
        [~, scores_gb] = predict(model.algorithm.gb_model, X_matrix);
        pred_gb = double(scores_gb(:,2) >= 0.5);
        
        [~, scores_rus] = predict(model.algorithm.rusboost_model, X_matrix);
        pred_rus = double(scores_rus(:,2) >= 0.5);
        
        nn_scores = model.algorithm.nn_model(X_norm')';
        pred_nn = double(nn_scores >= 0.5);
        
        % Votación mayoritaria
        vote_sum = pred_rf + pred_gb + pred_rus + pred_nn;
        predictions = double(vote_sum >= 2);
        
        % Calcular probabilidad como promedio
        all_scores = [scores_rf(:,2), scores_gb(:,2), scores_rus(:,2), nn_scores];
        probabilities = mean(all_scores, 2);
    catch predict_error
        error('Error al realizar predicciones con Ensamble por Votación: %s', predict_error.message);
    end
end

% Función auxiliar para verificar campos faltantes
function check_missing_fields(model, new_data)
    % Verificar si los datos son una tabla
    if ~istable(new_data)
        error('Los datos de entrada deben ser una tabla');
    end
    
    % Verificar campos requeridos
    if isfield(model.web_api, 'required_fields') && ~isempty(model.web_api.required_fields)
        missing_fields = model.web_api.required_fields(~ismember(model.web_api.required_fields, new_data.Properties.VariableNames));
        
        if ~isempty(missing_fields)
            error('Faltan campos requeridos: %s', strjoin(missing_fields, ', '));
        end
    end
end

