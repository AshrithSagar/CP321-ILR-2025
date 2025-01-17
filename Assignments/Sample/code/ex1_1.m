%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 1: Compute optimal trajectory  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('ex1_1.m'));
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
end



