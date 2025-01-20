%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Question 1: Compute closed-form trajectory  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('ex1_1_cubictraj.m'));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-robot-simulation')));

% Define time vector
time = linspace(0, 5, 50);

% Create robot from the custom RobotisWrapper class
robot = RobotisWrapper();

% Define initial position and target position by sampling two points from
% the workspace of the robot. 
% If you want to manually set them, make sure they are in the workspace,
% and use column vector notation.
initialPosition = robot.sampleRandomPosition();
targetPosition = robot.sampleRandomPosition();

% Compute trajectory based on third order polynomial
cartesianTrajectory = zeros(3, length(time));

%% ------ Write your code below ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %%

% Fill the 'trajectory' array with 3D position as column vectors
% The array 'trajectory' should start at 'initialPosition' 
% and end at 'targetPosition'.


%  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ %%
%% ------ Write your code above ------

% Plot resulting trajectory
figure;
plot3(cartesianTrajectory(1,:), cartesianTrajectory(2,:), cartesianTrajectory(3,:))
axis equal; grid on;
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
title('Closed-form time-dependent trajectory based on third-order polynomial');
legend('Trajectory');
disp("Press space to continue..."); pause();
close;

% Animate trajectory with robot
jointTrajectory = robot.computeInverseKinematics(cartesianTrajectory);
robot.animateTrajectory(jointTrajectory, cartesianTrajectory, targetPosition, 'Polynomial trajectory');
