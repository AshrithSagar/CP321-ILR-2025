%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 1: Compute optimal trajectory  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('ex1_2.m'));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-robot-simulation')));

% Create robot from the custom RobotisWrapper class
robot = RobotisWrapper();

optimal_control = MPC4DOF(robot);
target_position = [0.1; -0.3; 0.1];

%% Task 1: Minimum time
start = tic;
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumTime;
optimalSolution = optimal_control.solveOptimalTrajectory(target_position);
toc(start);
optimal_control.showResults(optimalSolution, target_position, 'Minimal Time');
disp("Press space to continue..."); pause();

%% Task 2: Minimum Cartesian distance
start = tic;
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumTaskDistance;
optimalSolution = optimal_control.solveOptimalTrajectory(target_position);
toc(start);
optimal_control.showResults(optimalSolution, target_position, 'Minimal Task Distance');
disp("Press space to continue..."); pause();

%% Task 3: Minimum joint distance
start = tic;
optimal_control.nlSolver.Optimization.CustomCostFcn = @minimumJointDistance;
optimalSolution = optimal_control.solveOptimalTrajectory(target_position);
toc(start);
optimal_control.showResults(optimalSolution, target_position, 'Minimal Joint Distance');

%% %%%%%%%%%%%%% User defined cost functions %%%%%%%%%%%%% %%
%% ------ Write your code below for Question 1 ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %%

% The robot arm is modeled with the following state space representation:
% - X: state vector = 4 joint angles of the robot arm
% - U(1:4): input vector = 4 joint speed
% - U(5): final time of the trajectory (constant for all timesteps), at
% which the robot arm should reach a specified target

%% OVERWRITE COST VARIABLE IN EACH FUNCTION %%

% Task 1: Minimum time
% This function minimizes the time scaling parameter Tf to minimize
% trajectory time
function cost = minimumTime(X, U, e, data, robot, target)

cost = 0;
for i = 1:data.PredictionHorizon
    Tf = U(5); % Final time of the trajectory
    cost = Tf; % Minimize the final time of the trajectory directly
end
end


% Task 2: Minimum distance in task space
% This function integrates dx = J*dq to minimize Cartesian trajectory length
% You can obtain the Jacobian J at configuration q using
% J = robot.fastJacobian(q)
% USE THE SQUARE OF THE NORM FOR NUMERICAL STABILITY
function cost = minimumTaskDistance(X, U, e, data, robot, target)

cost = 0;
for i = 1:data.PredictionHorizon
    q = X(i, :); % Joint positions at timestep i
    dq = U(i, 1:4)'; % Joint velocities at timestep i
    J = robot.fastJacobian(q'); % Jacobian at timestep i
    dx = J * dq; % Cartesian velocity
    cost = cost + norm(dx)^2; % Sum of squared Cartesian velocities
end
end


% Task 3: Minimum distance in joint space
% This function integrates dq to minimize joint trajectory length
% USE THE SQUARE OF THE NORM FOR NUMERICAL STABILITY
function cost = minimumJointDistance(X, U, e, data, robot, target)

cost = 0;
for i = 1:data.PredictionHorizon
    dq = U(i, 1:4)'; % Joint velocities at timestep i
    cost = cost + norm(dq)^2; % Sum of squared joint velocities
end
slack_penalty = norm(e)^2; % Penalize constraint violations
cost = cost + 100 * slack_penalty;
end
