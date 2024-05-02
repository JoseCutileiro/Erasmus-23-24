-module(train).
-compile(export_all).

% Check if a number is prime

is_prime(N) when N < 2 -> 
    false;
is_prime(2) -> true;
is_prime(3) -> true;
is_prime(N) when N > 3 -> 
    is_prime(N,2).

is_prime(N,Div) when N rem Div == 0 -> false;
is_prime(N,Div) when Div * Div > N -> true;
is_prime(N,Div) ->
    is_prime(N,Div + 1).


% Iota 

iota(N) -> lists:seq(1,N).

% Filter primes

filter_primes(L) -> 
    lists:filter(fun(X) -> is_prime(X) end, L).

% Mult by 2 (map)

mult2(L) -> 
    lists:map(fun(X) -> X * 2 end, L).

% Len

len(L) -> 
    len(L,0).

len([],C) -> C;

len([_|Tail], C) ->
    len(Tail,C + 1).