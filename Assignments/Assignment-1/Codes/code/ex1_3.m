%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 1: Initial trajectory  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('ex1_3.m'));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-robot-simulation')));


% Create robot and optimal control
robot = RobotisWrapper();
optimal_control = MPC4DOF(robot);

% define target position, maximal time and cost function for the solver
initial_joint_configuration = [0; 0; 0; 0];
target_position = [0.1; -0.3; 0.1];
max_time = 3;
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumJointDistance;

% find the optimal trajectory to the target
optimal_solution_full = optimal_control.solveOptimalTrajectory(target_position, ...
    initial_joint_configuration, max_time);
optimal_control.showResults(optimal_solution_full, target_position, 'Full trajectory');
disp("Press space to continue..."); pause(); close all;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 2: Add disturbance  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Replace ... with your code

disturbance_idx = round(length(optimal_solution_full.Topt) / 2);
% Mid-point of the trajectory

% Modify q_mid configuration to simulate a perturbation pushing the robot
% arm in another configuration
q_mid = optimal_solution_full.Xopt(:, disturbance_idx);
q_mid = q_mid + deg2rad([10; 0; 0; 0]);
% Add 10 degrees (converted to radians) to the first joint

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 3: Generate complete trajectory  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reallocate rest of time to still meet deadline
time_left = max_time - optimal_solution_full.Topt(disturbance_idx);

% Compute new trajectory starting from midpath joint state by providing
% joint state midpath as initial configuration for the solver
fprintf("--------- Reached intermediate target in %1.1f seconds --------- \n ", optimal_solution_full.Topt(disturbance_idx));
optimal_solution_after_disturbance = optimal_control.solveOptimalTrajectory(target_position, q_mid, time_left);

% Stitch trajectories together
%% ------ Write your code below ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %%

optimal_solution_final = [];
optimal_solution_final.Topt = [optimal_solution_full.Topt(1:disturbance_idx), ...
    optimal_solution_after_disturbance.Topt + optimal_solution_full.Topt(disturbance_idx)];
optimal_solution_final.Xopt = [optimal_solution_full.Xopt(:, 1:disturbance_idx), ...
    optimal_solution_after_disturbance.Xopt];
optimal_solution_final.Yopt = [optimal_solution_full.Yopt(:, 1:disturbance_idx), ...
    optimal_solution_after_disturbance.Yopt];
optimal_solution_final.MVopt = [optimal_solution_full.MVopt(:, 1:disturbance_idx), ...
    optimal_solution_after_disturbance.MVopt];

%  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ %%
%% ------ Write your code above ------

% Plot complete trajectory
f = figure('Name', 'Full trajectory with disturbance');
robot.animateTrajectory(optimal_solution_final.Xopt(:, 1:disturbance_idx), ...
    optimal_solution_final.Yopt(:, 1:disturbance_idx), target_position, 'Full trajectory with disturbance', f);
pause(0.3);
plot3([optimal_solution_full.Yopt(1, disturbance_idx) optimal_solution_after_disturbance.Yopt(1,1)], ...
    [optimal_solution_full.Yopt(2, disturbance_idx) optimal_solution_after_disturbance.Yopt(2,1)], ...
    [optimal_solution_full.Yopt(3, disturbance_idx) optimal_solution_after_disturbance.Yopt(3,1)], ...
    'color', 'r', 'LineWidth', 4);
pause(0.3);
robot.animateTrajectory(optimal_solution_final.Xopt(:, disturbance_idx:end), ...
    optimal_solution_final.Yopt(:, disturbance_idx:end), target_position, 'Full trajectory with disturbance', f);


%% Cost functions
%% ------ Write your code below for Question 1 ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %%

% Minimum distance in joint space
% This function integrate dq = u(1:4) to minimize joint trajectory length
function cost = minimumJointDistance(X, U, e, data, robot, target)

cost = 0;
for i = 1:data.PredictionHorizon
    dq = U(i, 1:4)'; % Joint velocities at timestep i
    cost = cost + norm(dq)^2; % Minimize sum of squared joint velocities
end
slack_penalty = norm(e)^2; % Penalize constraint violations
cost = cost + 100 * slack_penalty;
end
