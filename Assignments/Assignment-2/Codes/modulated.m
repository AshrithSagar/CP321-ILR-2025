%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%% %%%%%%%%%%% Step 0: Add library %%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('modulated.m'));
cd(filepath);

%% %%%%%%%%%%% Step 1: Define grid %%%%%%%%%%%%%%%%%%%%%%%%%%%
x_limits = [-3, 3];
y_limits = [-3, 3];
nb_gridpoints = 50;

% mesh domain
[x, y] = meshgrid(linspace(x_limits(1), x_limits(2), nb_gridpoints), ...
    linspace(y_limits(1), y_limits(2), nb_gridpoints));

%% %%%%%%%%%%% Generate a DS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ------ Write your code below ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %

M = [1, -2; 0, 1]; % Modulation matrix
a_1 = 1;
a_2 = -1;
A = diag([a_1, a_2]);

system = @(x) M * A * x;
[X_grid, Y_grid] = meshgrid(linspace(x_limits(1), x_limits(2), nb_gridpoints), ...
    linspace(y_limits(1), y_limits(2), nb_gridpoints));

x_dot = zeros(size(X_grid));
y_dot = zeros(size(Y_grid));

for i = 1:nb_gridpoints
    for j = 1:nb_gridpoints
        x_point = [X_grid(i, j); Y_grid(i, j)];
        vel = system(x_point);  % Compute velocity at each grid point
        x_dot(i, j) = vel(1);
        y_dot(i, j) = vel(2);
    end
end

%  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ &
%% ------ Write your code above ------

%% %%%%%%%%%%% Calculate path integral %%%%%%%%%%%%%%%%%%%%%%%

dt = 0.05;
iter = 0;
max_iter = 1000;

%% ------ Write your code below ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %

initial_position = [1; 1];
path_integral = initial_position; % path integral is intially a 2x1 array that will store the path and grow to a 2xN array.
% TODO: implement breaking conditions for while loop
while iter < max_iter
    % Velocity at the current position (interpolation on the grid)
    x_pos = path_integral(1, end);
    y_pos = path_integral(2, end);
    
    % Ensure the position is within bounds (clamping)
    if x_pos < x_limits(1) || x_pos > x_limits(2) || y_pos < y_limits(1) || y_pos > y_limits(2)
        break;  % Stop if out of bounds
    end
    
    % Find the closest grid point to the current position
    [~, x_idx] = min(abs(x(:, 1) - x_pos));
    [~, y_idx] = min(abs(y(1, :) - y_pos));
    
    % Get the velocity at that grid point
    v_x = x_dot(x_idx, y_idx);
    v_y = y_dot(x_idx, y_idx);
    
    % Integrate to get the next position using Euler’s method
    new_position = path_integral(:, end) + dt * [v_x; v_y];
    
    % Append the new position to the path
    path_integral(:, end + 1) = new_position; %#ok<AGROW>
    
    iter = iter + 1;
end

x0 = [1; 1];  % initial condition for the path
titleName = 'Path Integral of Modulated DS';
x_target = [];  % No additional target in this case

% Use the plot_ds function
plot_ds(x, y, x_dot, y_dot, path_integral, x0, titleName, x_target);

% Plot the final path (for debugging if necessary)
figure;
plot(path_integral(1, :), path_integral(2, :), 'r-', 'LineWidth', 2);
xlabel('x');
ylabel('y');
title('Path Integral of Modulated Dynamical System');
grid on;
axis equal;

%  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ &
%% ------ Write your code above ------

%% %%%%%%%%%%%% Plot Resulting DS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the function plot_ds describe below
% plot_ds(...);


function plot_ds(x, y, x_dot, y_dot, path, x0, titleName, x_target)

%   PLOT_DS  Plot a dynamical system on a grid.
%   PLOT_DS(X, Y, X_DOT, Y_DOT, PATH, X0, TITLENAME, X_TARGET) where the
%   arrays X,Y define the coordinates for X_DOT,Y_DOT and are monotonic and
%   2-D plaid (as if produced by MESHGRID), plots a dynamical system with
%   attractor(s) X0 given as 2xN vector data and name TITLENAME.
%
%   The optional variable X_TARGET given as 2xN vector data can be used to
%   plot additional points of interest (e.g. local modulation points).

hold on;

[~, h] = contourf(x, y, sqrt(x_dot.^2 + y_dot.^2), 80);
set(h, 'LineColor', 'none');
colormap('summer');
c_bar = colorbar;
c_bar.Label.String = 'Absolute velocity';
c_bar.Label.Interpreter = 'Latex';
c_bar.FontSize = 15;
c_bar.Label.FontSize = 20;

% Plot velocity stream
h_stream = streamslice(x, y, x_dot, y_dot, 2, 'method', 'cubic');
set(h_stream, 'LineWidth', 1);
set(h_stream, 'color', [0. 0. 0.]);
set(h_stream, 'HandleVisibility', 'off');
axis equal;

scatter(x0(1, :), x0(2, :), 100, 'r*', 'LineWidth', 2);

plot(path(1,:), path(2,:), 'r', 'LineWidth', 3);

if exist('x_target', 'var') && ~isempty(x_target)
    scatter(x_target(1, :), x_target(2, :), 100, 'bd', 'LineWidth', 2);
end

box on;
ax = gca;
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15;
xlabel('$x_1$', 'Interpreter', 'LaTex', 'FontSize', 20);
ylabel('$x_2$', 'Interpreter', 'LaTex', 'FontSize', 20);
title(titleName, 'Interpreter', 'LaTex', 'FontSize', 20);
end
