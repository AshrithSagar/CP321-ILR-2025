%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%%  Create solver
clear; close all; clc;
filepath = fileparts(which('ex1_4_inverse_kinematic.m'));
addpath(genpath(fullfile(filepath,'..', 'libraries', 'book-robot-simulation')));

robot = RobotisWrapper();

% Experiment parameters
targetPosition = [0.25; 0.2; 0];
maxJointSpeed = 0.3;
toleranceDistance = 1e-3;
%% Compute trajectories

% Pre-initialize dataset variable 'trajectories'
nTraj = 1;      % Number of trajectories
nPoints = 20;   % Number of points per trajectory
trajectories = nan(8, nPoints, nTraj); % Stores joint position and velocity
for iTraj=1:nTraj
    
    % Start at random configuration in the workspace
    q0 = robot.sampleRandomConfiguration();
    trajectory = []; % 8xN array with 4 joints position and 4 joints speed
    
    %% ------ Write your code below ------
    %  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %%

    % Compute one 'trajectory' from 'q0' to 'targetPosition'
    



    %  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ %%
    %% ------ Write your code above ------

    % Sub sample trajectory to a fixed-size array, and store it in
    % 'trajectories' dataset
    trajectorySubSampled = interp1(1:size(trajectory, 2), trajectory', linspace(1, size(trajectory, 2), nPoints))';
    trajectories(:, :, iTraj) = trajectorySubSampled;   

    % Show trajectory animation
    robot.animateTrajectory(trajectorySubSampled(1:4,:), robot.computeForwardKinematics(trajectorySubSampled(1:4,:)), targetPosition, 'inverse kinematic solution')
    disp('Press space to continue...'); pause();
    close(figure(1));
end


%% Display trajectories dataset
close all;
figure(2)
show(robot.robot, [robot.computeInverseKinematics(targetPosition, zeros(6,1), false); 0; 0], ...
            'PreservePlot', false, 'Frames', 'off', 'Visuals', 'on');
hold on;
plot3(targetPosition(1), targetPosition(2), targetPosition(3), 'ro', 'LineWidth', 1.5, 'MarkerSize', 15);

for iTraj = 1:nTraj
    cartesianPosition = robot.computeForwardKinematics(trajectories(1:4,:,iTraj));
    plot3(cartesianPosition(1,:), cartesianPosition(2,:), cartesianPosition(3,:), 'blue', 'LineWidth', 1.5);
end
view([45, 45])
axis([-0.3 0.4 -0.35 0.35 -0.2 0.5])
