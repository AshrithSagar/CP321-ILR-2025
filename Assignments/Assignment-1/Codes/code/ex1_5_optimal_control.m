%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%%  Create solver
clear; close all; clc;
filepath = fileparts(which('ex1_5_optimal_control.m'));
addpath(genpath(fullfile(filepath, '..', 'libraries', 'book-robot-simulation')));

robot = RobotisWrapper();
optimalControl = MPC4DOF(robot);
optimalControl.nlSolver.Optimization.CustomCostFcn = @myCost;

target_position = [0.25; 0; 0];
toleranceDistance = 10e-3;
maxTime = 5;
%% Generate batch of minimum time trajectories
nTraj = 10;
nPoints = optimalControl.nlSolver.PredictionHorizon + 1;
optimalTrajectories = nan(3, nPoints, nTraj);

h = waitbar(0,'Computing trajectories...');
for iTraj=1:nTraj
    
    % Find solution starting at random configuration
    q0 = robot.robot.randomConfiguration;
    % Last two parameters are used to speed up computation
    optimalSolution = optimalControl.solveOptimalTrajectory(target_position, q0, maxTime, true, true);
    
    % If the target is reached, append it to the dataset
    if norm(optimalSolution.Yopt(:, end) - target_position) < toleranceDistance
        optimalTrajectories(:,:,iTraj) = optimalSolution.Yopt;
    end
    
    % Visualize progression
    waitbar(iTraj/nTraj)
end
close(h)

% Display all successful trajectories
optimalControl.showTaskVolume(optimalTrajectories)

%% %%%%%%%%%%%%% User defined cost functions %%%%%%%%%%%%% %%
% minimumJointDistance
function cost = myCost(X, U, e, data, robot, target)

cost = 0;
for i = 1:data.PredictionHorizon
    dq = U(i, 1:4)'; % Joint velocities at timestep i
    cost = cost + norm(dq)^2; % Minimize sum of squared joint velocities
end
slack_penalty = norm(e)^2; % Penalize constraint violations
cost = cost + 100 * slack_penalty;
end
