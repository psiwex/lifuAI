clear; clc;
close all;

% Define grid parameters
%cart_data=sensor.mask;

axial_length = 65e-3;      % 65 mm focal length
Nx = 650;                  % Grid points (10 points/mm resolution)
dx = axial_length/Nx;      % Spatial step [m]
kgrid = kWaveGrid(Nx, dx);
%timeBar= 0:1e-8:5e-3; 
timeBar=0:.005:90;
kgrid.t_array = timeBar;  % Time array (5 ms pulse width)

% Medium properties
medium.sound_speed = 1540;    % Speed of sound in brain [m/s]
medium.density = 1000;         % Brain density [kg/m^3]
medium.BonA = 5;               % Nonlinearity parameter (typical for tissue)
medium.alpha_coeff = 0.75;     % Attenuation [dB/(MHz^y cm)] 
medium.alpha_power = 1.1;      % Power law exponent (adjust for skull+brain)

% Define Gaussian source (70mm diameter)
source_diameter = 70e-3;       % Transducer diameter [m]
sigma = source_diameter / 4;   % Gaussian width parameter
source_mask = exp(-((kgrid.x - 0).^2) ./ (2*sigma^2)); % 1D Gaussian profile

gauss=source_mask;
%source.p_mask = source_mask;

timeBar2=linspace(min(kgrid.x),max(kgrid.x),length(kgrid.t_array));
source_mask2 = exp(-((timeBar2 - 0).^2) ./ (2*sigma^2));

threshold = 0.5; % Adjust threshold for active regions

source.p_mask = gauss > threshold;
%source.p_mask = gauss >= 1;

% Pulse characteristics
source.p = 0.712e6;            % Peak pressure (Pr.0 = 0.712 MPa) [Pa]
source_freq = 1/(5e-3/10);     % Frequency = 10 cycles / 5ms = 2 kHz (adjust as needed)
source_freq=650000;
sourceGauss=source.p*source_mask2;
source_waveform = source.p * sin(2*pi*source_freq*kgrid.t_array);
%source_waveform=abs(source_waveform);
source.p = source_waveform; % 5ms pulse


% trying pure gauss
% source.p = sourceGauss;

% Create skull mask (3 mm thickness at start of grid)
skull_thickness = 3e-3;        % [m]
skull_mask = kgrid.x < skull_thickness;

% Apply skull attenuation (2 dB/mm = 200 dB/m)
skull_alpha = 200;             % [dB/m]
medium.alpha_coeff(skull_mask) = skull_alpha / (1e6^medium.alpha_power); 

sensor.mask = [false(1, Nx-1), true]; % Sensor at focal point (65mm)
sensor.record = {'p', 'p_max', 'p_rms'};

% Debugging
% sensor.mask = zeros(1, Nx);
% sensor.mask(end) = 1;  % Place sensor at the final grid point

num_sources = sum(source.p_mask(:));
source_waveform=source.p;
source.p = repmat(source_waveform, num_sources, 1);

%source_waveform = source.p * sin(2*pi*source_freq*kgrid.t_array);
%source_waveform = source_waveform .* (kgrid.t_array <= 5e-3); % Ensure correct length
sensor.mask=sensor.mask';

disp(size(source.p_mask));
disp(size(source.p));
disp(length(kgrid.t_array));

sensor_data = kspaceFirstOrder1D(kgrid, medium, source, sensor,'PlotLayout',false);

% From ISPTA.3 (716 mW/cm² = 7160 W/m²)
Q = 7160;                       % Volumetric heating rate [W/m²]
absorption_coeff = 2 * medium.alpha_coeff(end) * 1e-1; % Convert to Np/m
Q_volumetric = Q * absorption_coeff; % [W/m³]
peak_heating_rate = Q_volumetric * 1e-9; % [kW/cm³]

% Thermal parameters
rho = 1000;                     % Density [kg/m³]
c_p = 4200;                     % Specific heat [J/kg·K]
sonication_duration = 30;       % [s]

% Temperature rise (ΔT = Q * t / (ρ * c_p))
delta_T = (Q_volumetric * sonication_duration) / (rho * c_p); % [K]

% Output results
focal_pressure_mPa = sensor_data.p_max(end) / 1e4; % [mPa]
fprintf('Focal Pressure: %.2f MPa\n', focal_pressure_mPa);
fprintf('Peak Heating Rate: %.4f kW/cm³\n', peak_heating_rate);
fprintf('Temperature Rise: %.2f °C\n', delta_T);

% plot results
figure;
xLabels=linspace(0,65,length(sensor_data.p));
plot(xLabels,sensor_data.p(1, :), 'b-');
axis tight;
ylabel('Pressure (kPa)');
xlabel('Distance (mm)')
