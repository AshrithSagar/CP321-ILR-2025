%% Adapted from Learning Adaptive and Reactive Control for Robots, Aude Billard, LASA, EPFL %%
%% %%%%%%%%%%% Step 0: Add library %%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
filepath = fileparts(which('pendulum.m'));
cd(filepath);

%% %%%%%%%%%%% Step 1: Define grid %%%%%%%%%%%%%%%%%%%%%%%%%%%
x_limits = [-10, 10];
y_limits = [-10, 10];
nb_gridpoints = 50;

% mesh domain
[x, y] = meshgrid(linspace(x_limits(1), x_limits(2), nb_gridpoints), ...
    linspace(y_limits(1), y_limits(2), nb_gridpoints));

%% %%%%%%%%%%% Generate a DS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ------ Write your code below ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %

g = 9.81; % gravity
x_dot = y; % calculate velocity in x direction at each gridpoint

damping_choice = input('Do you want damping? (y/n): ', 's');
if strcmpi(damping_choice, 'y')
    % With damping
    y_dot = -g * sin(x) - y;
else
    % Without damping
    y_dot = -g * sin(x);
end

% Calculate absolute velocity at each gridpoint
abs_vel = sqrt(x_dot.^2 + y_dot.^2);

%  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ &
%% ------ Write your code above ------

%% %%%%%%%%%%% Calculate path integral %%%%%%%%%%%%%%%%%%%%%%%

dt = 0.05;
iter = 0;
max_iter = 1000;

%% ------ Write your code below ------
%  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv %

tol = 1e-3;
initial_position = [0, 0; 0.78, 0; 2.35, 0; 3.14, 1; 3.14, 4]';
path_integral = initial_position; % path integral is intially a 2x1 array that will store the path and grow to a 2xN array.
% TODO: implement breaking conditions for while loop
while true
    % TODO: integrate DS to get next position of the path integral
    current_pos = path_integral(:, end);
    
    vel_x = interp2(x, y, x_dot, current_pos(1), current_pos(2), 'linear', 0);
    vel_y = interp2(x, y, y_dot, current_pos(1), current_pos(2), 'linear', 0);
    
    next_pos = current_pos + dt * [vel_x; vel_y];
    
    path_integral(:, end+1) = next_pos;
    iter = iter + 1;
    
    if norm([vel_x; vel_y]) < tol || iter >= max_iter
        break;
    end
end

x0 = [0, 0; 0.78, 0; 2.35, 0; 3.14, 1; 3.14, 4]';
titleName = 'Path Integral';
x_target = [];

figure;
plot_ds(x, y, x_dot, y_dot, path_integral, x0, titleName, x_target);

figure;
hold on;
for i = 1:size(x0, 2)
    plot(path_integral(1, :), path_integral(2, :), 'r-', 'LineWidth', 2);
end
xlabel('x');
ylabel('y');
title('Path Integral');
grid on;
axis equal;
hold off;

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
