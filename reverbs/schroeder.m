%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% Schroeder Reverberator
% 
% Author: Chad McKell
% Date: 9 May 2020
% Place: University of California San Diego
%
% Description: The Schroeder reverberator is built using a combination of
% feedback comb filters and allpass filters
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
clear;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Define parameters
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Global parameters
fs = 44100;     % sampling frequency (samples/second)
Mmax = fs/10;   % delay line length (samples)
gamma = 0.001;  % tunable decay factor [see Jot paper]
T60 = 0.8;     	% reverberation time (seconds)

% Comb filter parameters 
M1 = 1300;              % delay length 1 (samples)
M2 = 1401;              % delay length 2 (samples)
M3 = 2251;              % delay length 3 (samples)   
g1 = 0.001^(M1/fs/T60); % feedback coeff 1                       
g2 = 0.001^(M2/fs/T60); % feedback coeff 2    
g3 = 0.001^(M3/fs/T60); % feedback coeff 3   

% Allpass filter parameters 
M4 = 347;               % delay length 4 (samples)
M5 = 113;               % delay length 5 (samples)
g4 = 0.7;               % feedback coeff 4
g5 = 0.7;               % feedback coeff 5

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Ensure that the feedback comb filter delay lengths are mutually prime
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if gcd(M1,M2) ~= 1
    error('All delay lengths must be mutually prime. M1 and M2 are not.')
end

if gcd(M1,M3) ~= 1
    error('All delay lengths must be mutually prime. M1 and M3 are not.')
end

if gcd(M2,M3) ~= 1
    error('All delay lengths must be mutually prime. M2 and M3 are not.')
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Define the input signal
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[x, fs] = audioread('guitar.wav'); 

% If user inserts stereo audio, only read the left channel 
if size(x,2) > 1
    x = x(:,1); 
    warning('File x has two channels. Only left channel was read.');
end

% Set the signal length and time bin vector 
N = length(x);          % signal length (samples)
nT = (0:N-1)/fs;        % time bin vector (seconds)   

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Pass the input signal through feedback comb filters connected in parallel
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y1 = feedbackcomb(x, N, Mmax, M1, g1);
y2 = feedbackcomb(x, N, Mmax, M2, g2);
y3 = feedbackcomb(x, N, Mmax, M3, g3);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Multiple each output by its corresponding comb filter delay length to  
% make all output spectrums have similar magnitude
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y = M1*y1 + M2*y2 + M3*y3;

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Pass the output from the feedback comb filters to allpass filters
% connected in series
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
y = allpass(y, N, Mmax, M4, g4);
y = allpass(y, N, Mmax, M5, g5);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Listen to the output signal
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
soundsc(y,fs);


