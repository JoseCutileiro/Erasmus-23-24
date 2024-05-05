-module(primes).
-compile(export_all).

is_prime(N) when N =< 1 -> false;
is_prime(2) -> true;
is_prime(N) ->
    is_prime(N, 2).

is_prime(N, Divisor) when Divisor * Divisor =< N ->
    if
        N rem Divisor =:= 0 -> false;
        true -> is_prime(N, Divisor + 1)
    end;
is_prime(_, _) -> true.


% simple map on array

prime_bool_array(Numbers) ->
    lists:map(fun is_prime/1, Numbers).


% with pmap

pmap(F, L) ->
  Parent = self(),
  [receive {Pid, Result} -> Result end || Pid <- [spawn(fun() -> Parent ! {self(), F(X)} end) || X <- L]].

prime_bool_array_par(Numbers) ->
    pmap(fun is_prime/1, Numbers).

% with a matrix

prime_bool_matrix_par(Matrix) ->
    lists:map(fun(Row) -> pmap(fun is_prime/1, Row) end, Matrix).