-module(foo).
-compile(export_all).

% quick sort erlang impl

qsort([]) -> 
    [];
qsort([X|Xs]) -> 
    qsort([Y || Y <- Xs, Y<X]) ++
    [X] ++
    qsort([Y || Y <- Xs, Y>= X]).


% list init

iota(N) -> lists:seq(1,N).

random_list(N) -> [rand:uniform(1000) || _ <- lists:seq(1,N)].

% benchmarking

benchmark(Fun,L) -> 
    NRuns = 500, 
    {T,_} = timer:tc(fun() ->
            [?MODULE:Fun(L)
            || _ <- lists:seq(1,NRuns)],
            ok
        end),
    T / (1000 * NRuns).

% parallel sorting

% Parent => the pid of the process
% Parent ! => Send the result back to the parent
% receive => Wait for the result

psort([]) -> 
    [];

psort([X|Xs]) -> 
    Parent = self(),
    spawn_link(
        fun() -> 
            Parent !
            psort([Y || Y <- Xs, Y >= X])
        end),
    psort([Y || Y <- Xs, Y < X]) ++ 
          [X] ++ 
          receive Ys -> Ys end.


% after benchmark -> parallel is slower ????? 
% Problem : Controlling granularity

psort2(Xs) -> psort2(5,Xs).

psort2(0,Xs) -> qsort(Xs); 
psort2(_,[]) -> [];
psort2(Depth,[X|Xs]) -> 
    Parent = self(),
    spawn_link(fun() ->
        Parent!
            psort2(Depth-1, [Y || Y <- Xs, Y >= X])
        end),
    psort2(Depth-1, [Y || Y <- Xs, Y < X]) ++ 
                    [X] ++
                    receive Ys -> Ys end.

% despite being faster is not 100% correct
% foo:qsort(R) != foo:psort2(R)
% messages are getting in the wrong place

psort3(Xs) ->
    psort3(5,Xs).

psort3(0,Xs) ->
    qsort(Xs);

psort3(_,[]) -> [];

psort3(D,[X|Xs]) ->
    Parent = self(),
    Ref = make_ref(),
    spawn_link(fun() ->
        Parent ! {Ref,psort3(D-1,[Y || Y <- Xs, Y >= X])}
    end),
    psort3(D-1,[Y || Y <- Xs, Y < X]) ++
        [X] ++
        receive {Ref,Greater} -> Greater end.
