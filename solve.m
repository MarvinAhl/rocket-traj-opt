%% Trajectory optimization for simple rocket dynamics using direct collocation
% Author: Marvin Ahlborn
% Date: 2025-03-02
%
% Trapezoidal collocation is used to solve the optimal control problem of a
% simple 2d rocket (see dynamics.mlx), only powered by one throttleable
% thrust-vector engine. The resulting NLP was solved with fmincon and the
% gradient of the objective, the jacobian of the constraints and the
% hessian of the lagrangian supplied analytically.

clear variables

% Variables
param.m = 1000;  % Mass [kg]
param.L = 10;  % Length [m]
param.l = 5;  % Distance engine to com [m]
param.J = 1/12 * param.m * param.L^2;  % Moment of inertia [kg m^2]
param.Fmax = 30000;  % Max thrust [N]
param.thetamax = deg2rad(10);  % Max thrust angle [rad]
param.g = 9.81;  % Earth acc [m/s^2]

% Start and end time
t0 = 0;
tf = 10;  % Only initial guess, end time is variable

% Start and end states
x0 = [0; 180; -deg2rad(90); 0; -40; 0];
xf = [0; 5; 0; 0; 0; 0];

% Number of collocation sections
n = 30;
tmesh = linspace(t0, tf, n+1);

% Solve
[x, u, tf] = collocation(x0, xf, tmesh, param);

% Animate
animate(x, u, t0, tf, param);

%% ----------------------- Functions -----------------------------

% System dynamics (from dynamics.mlx)
function xdot = f(x, u, param)
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
    x5 = x(5);
    x6 = x(6);
    u1 = u(1);
    u2 = u(2);

    J = param.J;
    g = param.g;
    l = param.l;
    m = param.m;

    xdot = [x4;x5;x6;-(u1.*sin(u2+x3))./m;-g+(u1.*cos(u2+x3))./m;-(l.*u1.*sin(u2))./J];
end

% The gradient of the dynamics w.r.t. x and u
function [gradx, gradu] = gradf(x, u, param)
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
    x5 = x(5);
    x6 = x(6);
    u1 = u(1);
    u2 = u(2);

    J = param.J;
    l = param.l;
    m = param.m;

    grad = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-(u1.*cos(u2+x3))./m,-(u1.*sin(u2+x3))./m,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,-sin(u2+x3)./m,cos(u2+x3)./m,-(l.*sin(u2))./J,0.0,0.0,0.0,-(u1.*cos(u2+x3))./m,-(u1.*sin(u2+x3))./m,-(l.*u1.*cos(u2))./J],[6,8]);
    gradx = grad(:, 1:6);
    gradu = grad(:, 7:8);
end

% The 6 hessians of the dynamics function
function [hessf1, hessf2, hessf3, hessf4, hessf5, hessf6] = hessf(x, u, param)
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
    x5 = x(5);
    x6 = x(6);
    u1 = u(1);
    u2 = u(2);

    J = param.J;
    l = param.l;
    m = param.m;

    hessf1 = zeros(8, 8);
    hessf2 = zeros(8, 8);
    hessf3 = zeros(8, 8);
    hessf4 = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,(u1.*sin(u2+x3))./m,0.0,0.0,0.0,-cos(u2+x3)./m,(u1.*sin(u2+x3))./m,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-cos(u2+x3)./m,0.0,0.0,0.0,0.0,-cos(u2+x3)./m,0.0,0.0,(u1.*sin(u2+x3))./m,0.0,0.0,0.0,-cos(u2+x3)./m,(u1.*sin(u2+x3))./m],[8,8]);
    hessf5 = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-(u1.*cos(u2+x3))./m,0.0,0.0,0.0,-sin(u2+x3)./m,-(u1.*cos(u2+x3))./m,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-sin(u2+x3)./m,0.0,0.0,0.0,0.0,-sin(u2+x3)./m,0.0,0.0,-(u1.*cos(u2+x3))./m,0.0,0.0,0.0,-sin(u2+x3)./m,-(u1.*cos(u2+x3))./m],[8,8]);
    hessf6 = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-(l.*cos(u2))./J,0.0,0.0,0.0,0.0,0.0,0.0,-(l.*cos(u2))./J,(l.*u1.*sin(u2))./J],[8,8]);
end

% Solves the optimal constrol problem by transcribing it to an NLP problem
% via direct collocation. tmesh gives the spacing of the grid points.
% Returns state and control interpolation functions 
function [x, u, tf] = collocation(x0, xf, tmesh, param)
    n = length(tmesh) - 1;
    tf0 = tmesh(end);  % Final time tf0
    tmeshnorm = tmesh / tf0;
    h = tmeshnorm(2:end) - tmeshnorm(1:end-1);  % The time differences per section
    
    % Initial guess for the state vector x, just linearly interpolate
    % between x0 and xf
    xinitmat = zeros(n+1, 8);
    alpha = linspace(0, 1, n+1);
    for i = 1:n+1
        xv0 = [x0; param.Fmax/2; 0];
        xvf = [xf; param.Fmax/2; 0];
        xinitmat(i, :) = alpha(i) * xvf + (1 - alpha(i)) * xv0;
    end
    xinit = [reshape(xinitmat', 8*(n+1), 1); tf0];

    % Lower and upper bounds
    lb = [-75; 0; -pi; -1000; -1000; -10*pi; -1e-10; -param.thetamax];
    ub = [75; 200; pi; 1000; 1000; 10*pi; param.Fmax; param.thetamax];
    tflb = 0;
    tfub = 20;
    lbrep = [repmat(lb, n+1, 1); tflb];
    ubrep = [repmat(ub, n+1, 1); tfub];

    % Scaling vector for all decision variables x
    scale.x = abs(ubrep - lbrep);
    % Scaling for the objective function
    scale.f = (tfub - tflb) * param.Fmax;
    % Scaling for collocation constraints at one section
    scale.collc = abs((ub(1:6) - lb(1:6)));
    % Scaling for one set of boundary constraints
    scale.bndc = abs((ub(1:6) - lb(1:6)));

    xinitsc = xinit ./ scale.x;
    lbrepsc = lbrep ./ scale.x;
    ubrepsc = ubrep ./ scale.x;

    options = optimoptions('fmincon', ...
        'Display', 'iter', ...
        'MaxFunctionEvaluations', 10000000, ...
        'MaxIterations', 100000, ...
        'Algorithm', 'interior-point', ...
        'ScaleProblem', false, ...
        'SpecifyObjectiveGradient', true, ...
        'SpecifyConstraintGradient', true, ...
        'HessianFcn', @(xsc, lambda) collhessian(xsc, lambda, h, param, scale));
    xsolsc = fmincon(@(x) collobj(x, h, param, scale), xinitsc, [], [], [], [], lbrepsc, ubrepsc, @(x) collcon(x, x0, xf, h, param, scale), options);
    xsol = xsolsc .* scale.x;

    x = @(t) statesplinesqr(xsol, tmeshnorm, t, param);
    u = @(t) controlsplinelnr(xsol, tmeshnorm, t, param);
    tf = xsol(tfidx(n));
end

% The optimization objective function for fmincon
function [fsc, gradsc] = collobj(xsc, h, param, scale)
    x = xsc .* scale.x;
    n = length(h);

    % Trapezoidal integration of the Lagrange objective
    % Just the thrust force integrated over time
    obj = 0;
    for i = 1:n
        ui = x(ithu(i));
        uip1 = x(ithu(i+1));
        tfi = x(tfidx(n));
        hi = h(i);

        obj = obj + hi*tfi/2 * (uip1(1) + ui(1));
    end

    fsc = obj ./ scale.f;  % Scaling
    gradsc = collobjgrad(xsc, h, param, scale);
end

% Supplies the analytic gradient of the objective function
function gradsc = collobjgrad(xsc, h, param, scale)
    x = xsc .* scale.x;
    n = length(h);
    grad = zeros(8 * (n+1) + 1, 1);

    tf = x(tfidx(n));

    % First index
    u1idx = ithu(1);
    u2idx = ithu(2);
    grad(u1idx(1), 1) = h(1) * tf / 2;

    % Stores the value of the derivative w.r.t. tf
    gradtfsum = h(1)/2 * (x(u1idx(1)) + x(u2idx(1)));
    for i = 2:n
        hi = h(i);
        him1 = h(i-1);

        uiidx = ithu(i);
        uip1idx = ithu(i+1);

        grad(uiidx(1), 1) = hi * tf / 2 + him1 * tf / 2;

        % Add elements to gradient w.r.t. tf
        gradtfsum = gradtfsum + hi/2 * (x(uiidx(1)) + x(uip1idx(1)));
    end

    % Last index
    unp1idx = ithu(n+1);
    grad(unp1idx(1), 1) = h(n) * tf / 2;

    % Gradient with respect to final time
    grad(tfidx(n), 1) = gradtfsum;

    % Scale
    gradsc = grad .* scale.x / scale.f;
end

% The constraints for fmincon
function [csc, ceqsc, jaccsc, jacceqsc] = collcon(xsc, x0, xf, h, param, scale)
    x = xsc .* scale.x;

    n = length(h);

    % No inequality constraints
    csc = [];
    jaccsc = [];

    % Equality constraints, 6*n collocation and 12 endpoint constraints
    ceqsc = zeros(6*n + 12, 1);
    % Collocation constraints
    for i = 1:n
        xi = x(ithx(i));
        xip1 = x(ithx(i+1));
        hi = h(i);
        ui = x(ithu(i));
        uip1 = x(ithu(i+1));
        tfi = x(tfidx(n));

        fi = f(xi, ui, param);
        fip1 = f(xip1, uip1, param);

        ceqcoll = xi - xip1 + hi*tfi/2 * (fi + fip1);  % Collocation constr
        ceqsc(ithcollc(i)) = ceqcoll ./ scale.collc;  % Scaling
    end

    % Endpoint constraints
    ceqstart = x(ithx(1)) - x0;
    ceqsc(startc(n)) = ceqstart ./ scale.bndc;

    ceqend = x(ithx(n+1)) - xf;
    ceqsc(endc(n)) = ceqend ./ scale.bndc;

    % Note how it has to be trasposed because matlab is weird about its
    % jacobian definition
    jacceqsc = collconjac(xsc, h, param, scale)';
end

% Supplies the analytic jacobian of the constraints for fmincon
function jacceqsc = collconjac(xsc, h, param, scale)
    x = xsc .* scale.x;
    n = length(h);

    jacceq = zeros(6*n + 12, 8 * (n+1) + 1);

    % Do jacobian calculation magic
    % Collocation constraints
    for i = 1:n
        xi = x(ithx(i));
        xip1 = x(ithx(i+1));
        ui = x(ithu(i));
        uip1 = x(ithu(i+1));
        hi = h(i);
        tf = x(tfidx(n));

        [gradxi, gradui] = gradf(xi, ui, param);
        % This second one is computed twice, here and in the next
        % iteration. Rescources could be saved by saving the value
        [gradxip1, graduip1] = gradf(xip1, uip1, param);

        fi = f(xi, ui, param);
        % Same here
        fip1 = f(xip1, uip1, param);

        % dci/dxi
        jacceq(ithcollc(i), ithx(i)) = eye(6) + hi * tf / 2 * gradxi;
        jacceq(ithcollc(i), ithu(i)) = hi * tf / 2 * gradui;
        jacceq(ithcollc(i), ithx(i+1)) = -eye(6) + hi * tf / 2 * gradxip1;
        jacceq(ithcollc(i), ithu(i+1)) = hi * tf / 2 * graduip1;
        jacceq(ithcollc(i), tfidx(n)) = hi/2 * (fi + fip1);
    end

    % End point constraints
    jacceq(startc(n), ithx(1)) = eye(6);
    jacceq(endc(n), ithx(n+1)) = eye(6);

    % Scale
    cscale = [repmat(scale.collc, n, 1); scale.bndc; scale.bndc];
    jacceqsc = jacceq .* scale.x' ./ cscale;
end

% Supplies the analytic hessian of the lagrangian for fmincon
function hessiansc = collhessian(xsc, lambda, h, param, scale)
    x = xsc .* scale.x;
    n = length(h);

    hessiansc = zeros(8 * (n+1) + 1, 8 * (n+1) + 1);

    % Hessian of objective
    objhess = zeros(8 * (n+1) + 1, 8 * (n+1) + 1);

    % i = 1 case
    h1 = h(1);
    u1idx = ithu(1);

    objhess(u1idx(1), tfidx(n)) = h1/2;
    objhess(tfidx(n), u1idx(1)) = h1/2;

    % i = 2 to n case
    for i = 2:n
        hi = (h(i));
        him1 = h(i-1);
        uiidx = ithu(i);

        objhess(uiidx(1), tfidx(n)) = 1/2 * (hi + him1);
        objhess(tfidx(n), uiidx(1)) = 1/2 * (hi + him1);
    end

    % i = n+1 case
    hn = h(n);
    unp1idx = ithu(n+1);

    objhess(unp1idx(1), tfidx(n)) = hn/2;
    objhess(tfidx(n), unp1idx(1)) = hn/2;

    % Add to output matrix
    hessiansc = hessiansc + objhess .* scale.x .* scale.x' ./ scale.f;

    % Hessian of collocation constraints
    for i = 1:n
        xi = x(ithx(i));
        ui = x(ithu(i));
        xip1 = x(ithx(i+1));
        uip1 = x(ithu(i+1));
        hi = h(i);
        tf = x(tfidx(n));

        [gradxi, gradui] = gradf(xi, ui, param);
        % This second one is computed twice, here and in the next
        % iteration. Rescources could be saved by saving the value
        [gradxip1, graduip1] = gradf(xip1, uip1, param);
        [hessfi1, hessfi2, hessfi3, hessfi4, hessfi5, hessfi6] = hessf(xi, ui, param);
        hessfi = zeros(8, 8, 6);
        hessfi(:, :, 1) = hessfi1;
        hessfi(:, :, 2) = hessfi2;
        hessfi(:, :, 3) = hessfi3;
        hessfi(:, :, 4) = hessfi4;
        hessfi(:, :, 5) = hessfi5;
        hessfi(:, :, 6) = hessfi6;
        % Same here I guess
        [hessfip11, hessfip12, hessfip13, hessfip14, hessfip15, hessfip16] = hessf(xip1, uip1, param);
        hessfip1 = zeros(8, 8, 6);
        hessfip1(:, :, 1) = hessfip11;
        hessfip1(:, :, 2) = hessfip12;
        hessfip1(:, :, 3) = hessfip13;
        hessfip1(:, :, 4) = hessfip14;
        hessfip1(:, :, 5) = hessfip15;
        hessfip1(:, :, 6) = hessfip16;
        
        for j = 1:6
            hessiancij = zeros(8 * (n+1) + 1, 8 * (n+1) + 1);

            % Compute the actual hessian
            % dcij/dxidxi
            hessiancij(ithx(i), ithx(i)) = hi * tf / 2 * hessfi(1:6, 1:6, j);
            % dcij/dxidui and reverse
            hessiancij(ithx(i), ithu(i)) = hi * tf / 2 * hessfi(1:6, 7:8, j);
            hessiancij(ithu(i), ithx(i)) = hi * tf / 2 * hessfi(7:8, 1:6, j);
            % dcij/duidui
            hessiancij(ithu(i), ithu(i)) = hi * tf / 2 * hessfi(7:8, 7:8, j);
            % dcij/dxidtf and reverse
            hessiancij(ithx(i), tfidx(n)) = hi/2 * gradxi(j, :)';
            hessiancij(tfidx(n), ithx(i)) = hi/2 * gradxi(j, :);
            % dcij/duidtf and reverse
            hessiancij(ithu(i), tfidx(n)) = hi/2 * gradui(j, :)';
            hessiancij(tfidx(n), ithu(i)) = hi/2 * gradui(j, :);

            % Same as before for xi+1 and ui+1
            % dcij/dxi+1dxi+1
            hessiancij(ithx(i+1), ithx(i+1)) = hi * tf / 2 * hessfip1(1:6, 1:6, j);
            % dcij/dxi+1dui+1 and reverse
            hessiancij(ithx(i+1), ithu(i+1)) = hi * tf / 2 * hessfip1(1:6, 7:8, j);
            hessiancij(ithu(i+1), ithx(i+1)) = hi * tf / 2 * hessfip1(7:8, 1:6, j);
            % dcij/dui+1dui+1
            hessiancij(ithu(i+1), ithu(i+1)) = hi * tf / 2 * hessfip1(7:8, 7:8, j);
            % dcij/dxi+1dtf and reverse
            hessiancij(ithx(i+1), tfidx(n)) = hi/2 * gradxip1(j, :)';
            hessiancij(tfidx(n), ithx(i+1)) = hi/2 * gradxip1(j, :);
            % dcij/dui+1dtf and reverse
            hessiancij(ithu(i+1), tfidx(n)) = hi/2 * graduip1(j, :)';
            hessiancij(tfidx(n), ithu(i+1)) = hi/2 * graduip1(j, :);
            
            % Scale it
            hessiancijsc = hessiancij .* scale.x .* scale.x' ./ scale.collc(j);
            % Multiply it by the appropriate lambda and add it to the total hessian
            hessiansc = hessiansc + lambda.eqnonlin(6 * (i-1) + j) * hessiancijsc;
        end
    end

    % Endpoint constraint hessians are just zero
end

% Functions for finding the right indices in the x and constraint vectors
function idx = ithx(i)
    idx = 8 * i - 7 : 8 * i - 2;
end

function idx = ithu(i)
    idx = 8 * i - 1 : 8 * i;
end

function idx = tfidx(n)
    % Actually just the last element but I feel like this is good practice
    idx = 8 * (n+1) + 1;
end

function idx = ithcollc(i)
    idx = 6 * (i-1) + 1 : 6 * (i-1) + 6;
end

function idx = startc(n)
    idx = 6*n + 1 : 6*n + 6;
end

function idx = endc(n)
    idx = 6*n + 7 : 6*n + 12;
end

% Functions for interpolating the final result
function xs = statesplinesqr(xmeshtf, tmeshnorm, ts, param)
    nt = length(ts);

    xmesh = xmeshtf(1:end-1);
    tmesh = tmeshnorm * xmeshtf(end);

    xs = zeros(nt, 6);  % 6 elements per state
    for i = 1 : nt
        % Find right after which grid point in tmesh ts comes
        meshidxs = find((tmesh - ts(i)) <= 0);
        meshidx = meshidxs(end);

        if meshidx == length(tmesh)
            xs(i, :) = xmesh(ithx(meshidx));
        else
            t0 = tmesh(meshidx);
            t1 = tmesh(meshidx + 1);
    
            x0 = xmesh(ithx(meshidx));
            x1 = xmesh(ithx(meshidx+1));
            u0 = xmesh(ithu(meshidx));
            u1 = xmesh(ithu(meshidx+1));
    
            xdot0 = f(x0, u0, param);
            xdot1 = f(x1, u1, param);
    
            % Make quadratic spline with x0, xdot0 at t0 and xdot1 at t1
            spline = @(t) ((xdot0-xdot1)./(t0.*2.0-t1.*2.0)).*t.^2 + ...
                          ((t0.*xdot1.*2.0-t1.*xdot0.*2.0)./(t0.*2.0-t1.*2.0)).*t - ...
                          (t0.*x0.*-2.0+t1.*x0.*2.0+t0.^2.*xdot0+t0.^2.*xdot1-t0.*t1.*xdot0.*2.0)./(t0.*2.0-t1.*2.0);
    
            % Evaluate spline at ts(i) and save to xs
            xs(i, :) = spline(ts(i));
        end
    end
end

function us = controlsplinelnr(xmeshtf, tmeshnorm, ts, param)
    nt = length(ts);

    xmesh = xmeshtf(1:end-1);
    tmesh = tmeshnorm * xmeshtf(end);

    us = zeros(nt, 2);
    for i = 1:nt
        % Find right after which grid point in tmesh ts comes
        meshidxs = find((tmesh - ts(i)) <= 0);
        meshidx = meshidxs(end);

        if meshidx == length(tmesh)
            us(i, :) = xmesh(ithu(meshidx));
        else
            t0 = tmesh(meshidx);
            t1 = tmesh(meshidx + 1);
    
            u0 = xmesh(ithu(meshidx));
            u1 = xmesh(ithu(meshidx+1));
    
            % Make linear spline with u0 at t0 and u1 at t1
            spline = @(t) ((u0-u1)./(t0-t1)).*t + ...
                          (t0.*u1-t1.*u0)./(t0-t1);
    
            % Evaluate spline at ts(i) and save to us
            us(i, :) = spline(ts(i));
        end
    end
end

%% ------------------- Animation functions ----------------------

% Rotate from body to inertial frame
function R = R_nb(phi)
    R = [cos(phi), -sin(phi); sin(phi), cos(phi)];
end

function animate(x, u, t0, tf, param)
    % Export as gif
    savegif = false;

    % Initialize variables
    fps = 24;
    frames = round((tf - t0) * fps);
    dt = (tf - t0) / frames;
    tspan = linspace(t0, tf, frames);

    % Obtain state and control data from interpolation functions
    xhistory = zeros(length(tspan), 6);
    uhistory = zeros(length(tspan), 2);
    for i = 1:length(tspan)
        xhistory(i, :) = x(tspan(i))';
        uhistory(i, :) = u(tspan(i))';
    end

    if savegif
        % Make the resulting animation a little nicer, only for this particular
        % simulation
        % Add a few falling frames in the beginning
        for i = 1:20
            frames = frames + 1;
            tspan(end+1) = tspan(end) + dt;
            xhistory = [xhistory(1, :) + [0, -xhistory(1, 5)*dt, 0, 0, 0, 0]; xhistory];
            uhistory = [[0, 0]; uhistory];
        end
    
        % Add a few more frames in the end to make it less abrupt
        for i = 1:10
            frames = frames + 1;
            tspan(end+1) = tspan(end) + dt;
            xhistory(end+1, :) = xhistory(end, :);
            uhistory(end+1, :) = [0, 0];
        end
    end

    % Animate
    fig = figure();
    fig.Position(2) = 50;
    fig.Position(3) = 600;
    fig.Position(4) = 700;
    
    % Draw ground
    yline(0, 'k-', 'LineWidth', 2);
    hold on
    
    % Draw rocket
    flamescale = 5;
    
    rcom = [xhistory(1, 1); xhistory(1, 2)];
    rtip = R_nb(xhistory(1, 3)) * [0; param.L-param.l] + rcom;
    reng = R_nb(xhistory(1, 3)) * [0; -param.l] + rcom;
    rflame = R_nb(xhistory(1, 3) + uhistory(1, 2)) * [0; -uhistory(1, 1) / param.Fmax * flamescale] + reng;
    
    rocket = plot([rtip(1) reng(1)], [rtip(2) reng(2)], 'k-', 'LineWidth', 4);
    flame = plot([reng(1) rflame(1)], [reng(2), rflame(2)], 'r-', 'LineWidth', 2);
    
    window_size_x = 75;
    window_size_yup = 200;
    window_size_ydown = 10;
    axis equal
    xlim([-window_size_x, window_size_x])
    ylim([-window_size_ydown, window_size_yup])
    grid on
    
    title("Rocket Simulation")
    xlabel("x / m")
    ylabel("y / m")
    
    % Animate rocket
    for frame = 1:frames
        pause(dt/2);
    
        rcom = [xhistory(frame, 1); xhistory(frame, 2)];
        rtip = R_nb(xhistory(frame, 3)) * [0; param.L-param.l] + rcom;
        reng = R_nb(xhistory(frame, 3)) * [0; -param.l] + rcom;
        rflame = R_nb(xhistory(frame, 3) + uhistory(frame, 2)) * [0; -uhistory(frame, 1) / param.Fmax * flamescale] + reng;
        
        rocket.XData = [rtip(1), reng(1)];
        rocket.YData = [rtip(2), reng(2)];
        flame.XData = [reng(1), rflame(1)];
        flame.YData = [reng(2), rflame(2)];
    
        axis equal
        xlim([-window_size_x, window_size_x])
        ylim([-window_size_ydown, window_size_yup])
    
        drawnow

        if savegif
            exportgraphics(gcf, 'rocket.gif', 'Append', true);
        end
    end
end