function [Best_Cost, Best_X, Convergence_curve] = CHMO(nP, MaxIt, lb, ub, dim, fobj)
% Chaos-Enhanced Harmonic Mean Optimizer (CHMO)
% Improved version with chaos theory by Dr. Pradeep Jangir (pkjmtech@gmail.com)

%% Initialization
Cost = zeros(nP, 1);
X = initialization(nP, dim, ub, lb);

% Initialize chaotic map (Logistic map)
chaos = zeros(MaxIt, 1);
chaos(1) = 0.7; % Initial chaotic value (between 0 and 1)
mu = 4; % Chaotic parameter (fully chaotic behavior)

for i = 1:nP
    Cost(i) = fobj(X(i, :));
end

[~, ind] = sort(Cost);
Best_X = X(ind(1), :);
Best_Cost = Cost(ind(1));
Worst_Cost = Cost(ind(end));
Worst_X = X(ind(end), :);
I = randi([2 5]);
Better_X = X(ind(I), :);
Better_Cost = Cost(ind(I));

%% Main Loop of CHMO
for it = 1:MaxIt
    % Update chaotic sequence
    if it > 1
        chaos(it) = mu*chaos(it-1)*(1-chaos(it-1));
    end
    
    % Chaotic adaptation of parameters
    alpha = 2 * exp(-4 * (it / MaxIt)) * (1 + 0.1*chaos(it));  % Chaotic alpha
    r_chaos = 0.1 + 0.4*chaos(it);  % Chaotic r parameter
    
    M_Best = Best_Cost;
    M_Better = Better_Cost;
    M_Worst = Worst_Cost;
    
    for i = 1:nP
        % Chaotic updating rules
        del = 2 * rand * alpha - alpha;  % Eq. (5)
        sigm = 2 * rand * alpha - alpha;  % Eq. (9)
        
        % Chaotic selection of solutions
        A1 = randperm(nP);
        A1(A1 == i) = [];
        a = A1(1); b = A1(2); c = A1(3);
        e = 1e-25;
        epsi = e * rand;
        
        % Compute harmonic mean of selected solutions
        X_sum = 1 ./ X([a, b, c], :);
        harmonic_mean = 3 ./ sum(X_sum, 1);
        
        % Compute differences from harmonic mean
        MM = [norm(X(a, :) - harmonic_mean), norm(X(b, :) - harmonic_mean), norm(X(c, :) - harmonic_mean)];
        omg = max(MM) + epsi;
        
        % Compute weights inversely proportional to differences
        W(1) = cos(MM(1) + pi) * exp(-abs(MM(1) / omg));  % Eq. (4.2)
        W(2) = cos(MM(2) + pi) * exp(-abs(MM(2) / omg));  % Eq. (4.3)
        W(3) = cos(MM(3) + pi) * exp(-abs(MM(3) / omg));  % Eq. (4.4)
        Wt = sum(W);
        
        % Compute weighted movement WM1 with chaotic factor
        WM1 = del * (W(1) * (X(a, :) - X(b, :)) + W(2) * (X(a, :) - X(c, :)) + ...
            W(3) * (X(b, :) - X(c, :))) / (Wt + epsi) * (1 + 0.1*chaos(it));
        
        % Global harmonic mean differences
        global_solutions = [Best_X; Better_X; Worst_X];
        X_sum_global = 1 ./ global_solutions;
        harmonic_mean_global = 3 ./ sum(X_sum_global, 1);
        
        MM_global = [norm(Best_X - harmonic_mean_global), norm(Better_X - harmonic_mean_global), norm(Worst_X - harmonic_mean_global)];
        omg_global = max(MM_global) + epsi;
        
        % Compute global weights with chaotic adaptation
        W(1) = cos(MM_global(1) + pi) * exp(-abs(MM_global(1) / omg_global)) * (1 + 0.05*chaos(it));
        W(2) = cos(MM_global(2) + pi) * exp(-abs(MM_global(2) / omg_global)) * (1 + 0.05*chaos(it));
        W(3) = cos(MM_global(3) + pi) * exp(-abs(MM_global(3) / omg_global)) * (1 + 0.05*chaos(it));
        Wt = sum(W);
        
        % Compute weighted movement WM2 with chaotic factor
        WM2 = del * (W(1) * (Best_X - Better_X) + W(2) * (Best_X - Worst_X) + ...
            W(3) * (Better_X - Worst_X)) / (Wt + epsi) * (1 + 0.1*chaos(it));
        
        % Chaotic MeanRule
        r = r_chaos; % Use chaotic r instead of uniform random
        MeanRule = r * WM1 + (1 - r) * WM2;  % Eq. (4)
        
        % Generate new solutions z1 and z2 with chaotic factors
        if rand < (0.5 + 0.1*chaos(it)) % Chaotic probability
            z1 = X(i, :) + sigm * (chaos(it) .* MeanRule) + randn .* (Best_X - X(a, :)) / (M_Best - Cost(a) + 1);
            z2 = Best_X + sigm * (chaos(it) .* MeanRule) + randn .* (X(a, :) - X(b, :)) / (Cost(a) - Cost(b) + 1);
        else  % Eq. (8)
            z1 = X(a, :) + sigm * (chaos(it) .* MeanRule) + randn .* (X(b, :) - X(c, :)) / (Cost(b) - Cost(c) + 1);
            z2 = Better_X + sigm * (chaos(it) .* MeanRule) + randn .* (X(a, :) - X(b, :)) / (Cost(a) - Cost(b) + 1);
        end
        
        % Vector combining stage with chaotic mutation
        u = zeros(1, dim);
        for j = 1:dim
            mu = 0.05 * randn * (1 + 0.1*chaos(it)); % Chaotic mutation
            if rand < (0.5 + 0.1*chaos(it)) % Chaotic probability
                if rand < 0.5
                    u(j) = z1(j) + mu * abs(z1(j) - z2(j));  % Eq. (10.1)
                else
                    u(j) = z2(j) + mu * abs(z1(j) - z2(j));  % Eq. (10.2)
                end
            else
                u(j) = X(i, j) * (1 + 0.05*chaos(it));  % Eq. (10.3) with chaotic perturbation
            end
        end
        
        % Chaotic local search stage
        if rand < (0.5 + 0.1*chaos(it)) % Chaotic probability
            L = rand < (0.5 + 0.1*chaos(it));
            v1 = (1 - L) * 2 * rand + L;  % Eq. (11.5)
            v2 = rand * L + (1 - L);  % Eq. (11.6)
            Xavg = harmonic_mean;  % Use harmonic mean instead of arithmetic mean  % Eq. (11.4)
            phi = rand * (1 + 0.1*chaos(it));
            Xrnd = phi * Xavg + (1 - phi) * (phi * Better_X + (1 - phi) * Best_X);  % Eq. (11.3)
            Randn = L * randn(1, dim) + (1 - L) * randn;
            if rand < 0.5
                u = Best_X + Randn .* (MeanRule + randn .* (Best_X - X(a, :)));  % Eq. (11.1)
            else
                u = Xrnd + Randn .* (MeanRule + randn .* (v1 * Best_X - v2 * Xrnd));  % Eq. (11.2)
            end
        end
        
        % Check if new solution goes outside the search space and bring it back
        New_X = BC(u, lb, ub);
        New_Cost = fobj(New_X);
        
        % Chaotic greedy selection
        if New_Cost < Cost(i) || chaos(it) > 0.9 % Occasionally accept worse solutions
            X(i, :) = New_X;
            Cost(i) = New_Cost;
            if Cost(i) < Best_Cost
                Best_X = X(i, :);
                Best_Cost = Cost(i);
            end
        end
    end
    
    % Determine the worst solution
    [~, ind] = sort(Cost);
    Worst_X = X(ind(end), :);
    Worst_Cost = Cost(ind(end));
    
    % Chaotic selection of better solution
    I = 2 + floor(chaos(it)*(nP-2));
    Better_X = X(ind(I), :);
    Better_Cost = Cost(ind(I));
    
    % Update Convergence_curve
    Convergence_curve(it) = Best_Cost;
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(Best_Cost)]);
end
end

% Boundary checking function
function X = BC(X, lb, ub)
Flag4ub = X > ub;
Flag4lb = X < lb;
X = (X .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
end

% Initialization function
function X = initialization(nP, dim, ub, lb)
X = rand(nP, dim) .* (ub - lb) + lb;
end