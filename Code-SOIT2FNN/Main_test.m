%% SOIT2FNN-MO test stage
% Please read the README file first before running the code.

%% Clear MATLAB cache
clc;
clear all;
close all;

%% 
I = 9;%12
M = 3;
%% Load weights and dataset
subfolder_data = 'IT2FNN_normalization_data';
subfolder_weights = 'Weights';
filename_weight_loading = fullfile(subfolder_weights, 'IT2FNN3_Chaotic_0.0_weights_I9_M3_C5_Tg0.002500_Tr0.002500.mat');
% filename_weight_loading = fullfile(subfolder_weights, 'IT2FNN3_prs_weights_I9_M3_C10_Tg0.001000_Tr0.001000.mat');
load(filename_weight_loading);

%For Chaotic time series
filename_test = fullfile(subfolder_data, 'Chaotic_test_9_3_0.000000.mat');
% filename_test = fullfile(subfolder_data, 'Chaotic_train_9_3_0.100000.mat');
load(filename_test)

% % %For microgrid time series
% filename_test = fullfile(subfolder_data, 'prs_HWM_test.mat');
% % filename_test = fullfile(subfolder_data, 'unmets_HWM_train.mat');
% load(filename_test)
% inputs = testData(:, 4:15);%testData(:, 1:15);

X_test = test_input';
Y_test = test_output';
% X_test = train_input';
% Y_test = train_output';


If_normalization = 0; % 1: Normalization' 0: not

[My_y] = IT2FNNP (X_test,ant_par,con_par,Mqlqr,I,Y_test,Mant, K_link,Mylyr);

% plot(My_y(:,1), 'r--', 'LineWidth', 2);
%% Visualization & Error

mse = mean((My_y - Y_test)'.^2, 2); 
disp('loss(mse): ');
disp(mse)

rmse = sqrt(mse);

disp('rmse:')
disp(rmse)

filename_raw = '2012_Initial_data - unmet.csv';
% filename_raw = '2012_Initial_data-pr.csv';
original_data = csvread(filename_raw);

min_vals = min(original_data);
max_vals = max(original_data);

if If_normalization == 1
Y_back = Y_test*(max_vals-min_vals) + min_vals;  
My_y_back = My_y*(max_vals-min_vals) + min_vals;
else
Y_back = Y_test;  
My_y_back = My_y;
end  

for iii = 1:M
  
    figure(iii+1)
    plot(Y_back(:,iii), 'b', 'LineWidth', 2);  
    hold on; 
    plot(My_y_back(:,iii), 'r--', 'LineWidth', 2);
    hold off;

end

% disp('mean(rmse):')
% disp(mean(rmse))

% %% SAVE For post_processing
% yp_mo = My_y_back;
% ya_mo = Y_back;
% 
% error_mo = ya_mo - yp_mo;
% r_error_mo = abs(ya_mo - yp_mo)./ya_mo;


% subfolder_weights = 'Post_processing';
% filename_results = fullfile(subfolder_weights, 'unmets_results_MO.mat');
% % filename_results = fullfile(subfolder_weights, 'Chaotic_results_MO.mat');
% save(filename_results, 'error_mo','r_error_mo','yp_mo','ya_mo');

