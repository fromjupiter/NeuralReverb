%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% Feedback Comb Filter 
% 
% Author: Chad McKell
% Date: 9 May 2020
% Place: University of California San Diego
%
% Description: This script implements a feedback comb filter using a 
% circular delay line.
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

function y = feedbackcomb(x, N, Mmax, M, g)

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Input parameters
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% x : input impulse signal  
% N : signal length (samples)
% Mmax : delay line length (samples)
% M : delay length (samples)
% g : feedback coefficient           

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Handle errors
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if M > Mmax-1
    error('M must be less than Mmax samples')
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Compute filtered signal using circular delay line
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Initialize delay line and output signal
delayLine = zeros(Mmax,1);
y = zeros(N,1);

% Initialize pointer to 'read' bin of delay line   
ip = 1;

for i = 1:N
    
    % Compute pointer to 'write' bin of delay line
    op = ip-floor(M); 

    % Handle case where 'op' is out of bounds
    if op < 1
        op = op + Mmax;
    end
    
    % Read in input signal to delay line at 'ip'
    delayLine(ip) = g * (x(i) + delayLine(op));
    
    % Find 'g * y(i-M)' 
    yM = delayLine(op);
    
    % Write to output signal 
    y(i) = x(i) + yM;
    
    % Increment read pointer
    ip = ip+1;
    
    % Handle case where 'ip' is out of bounds
    if ip > Mmax
       ip = ip - Mmax;
    end 
end

end

