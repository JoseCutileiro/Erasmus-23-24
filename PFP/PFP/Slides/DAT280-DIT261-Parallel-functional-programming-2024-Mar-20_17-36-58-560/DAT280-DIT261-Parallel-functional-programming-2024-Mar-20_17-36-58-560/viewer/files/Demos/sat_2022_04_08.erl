-module(sat).
-compile(export_all).



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SAT solver demo
%%

%% The problem to solve is represented as a list of lists 
%% of integers not equal to zero. 
%%
%% ex: [[1, 2, 3] [-1, 2, 3]] 
%% 
%% The numbers come from a numbering of the atoms involved a, b, c in
%% the boolean expression to attempt to satisfy. 
%%
%% Each list in the list of lists is called a clause and the boolean 
%% connective between elements in this list is disjunction. 
%% 
%% The boolean connective between each clause is conjunction. 
%% This representation is called "conjunctive normal form" and 
%% all boolean expressions can be converted to this format. 
%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sequential 
%%

solve([]) ->
    [];
solve([[]|_]) ->
    false;
solve([[L|Lits]|Clauses]) ->
    case solve(unit_propagation(L, Clauses)) of
        false ->
            case solve(unit_propagation(-L,[Lits|Clauses])) of
                false ->
                    false;
                Solution ->
                    [-L|Solution]
            end;
        Solution ->
            [L|Solution]
    end.

%% Unit propagation removes occurrences of the negation of a term
%% in all clauses, unless that clause contains the term itself 
%% any clause containing L and -L is satisfyable for any value assignment 
%% true, false to L. 
%% 

unit_propagation(L, Clauses) ->
    NewClauses = 
        [lists:delete(-L,C) || C <- Clauses,
                               not lists:member(L,C)],
    sort_by_length(NewClauses).

%% Shorter lists in the beginning
sort_by_length(Clauses) ->
    [C || {_,C} <- lists:usort([{length(C),C} || C <- Clauses])].


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Benchmarking
%%


%% Fetch hard example--80 variables, 344 clauses, takes 1 second to solve.

hard_example() ->
    {ok,[P]} = file:consult("formula.txt"),
    P.

%% Harder: 110 variables, 473 clauses, takes around 9 seconds to solve.
harder_example() ->
    {ok,[P]} = file:consult("harder.txt"),
    P.


benchmark(Name) ->
    N = 2,
    P = harder_example(),
    Times = [begin
		 {T,_} = timer:tc(fun() -> ?MODULE:Name(P) end),
		 io:format("."),
		 T
	     end
	     || _ <- lists:seq(1,N)],
    io:format("\n"),
    io:format("avg: ~f\n",[lists:sum(Times) / (N * 1000)]),
    io:format("min: ~f\n",[lists:min(Times) / 1000]),
    io:format("max: ~f\n",[lists:max(Times) / 1000]).

benchmarkem([]) ->
    ok;
benchmarkem([Solver|Solvers]) ->
    {Name, S} = Solver,
    io:format("Benchmarking: ~s\n", [Name]),
    benchmark(S),
    io:format("\n\n"),
    benchmarkem(Solvers).

benchmark_all() ->
    AllSolvers = [{"solve", solve}, 
                  {"par_solve", par_solve}, 
                  {"pool_solve", pool_solve},
                  {"limit_solve", limit_solve}],
    benchmarkem(AllSolvers).

                   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parallel 1
%% Speculative thread working in parallel on the negated "head-term".
%% 

par_solve([]) ->
    [];
par_solve([[]|_]) ->
    false;
par_solve([[L|Lits]|Clauses]) ->
    Rest = speculate(fun()-> par_solve(unit_propagation(-L,[Lits|Clauses])) end),
    case par_solve(unit_propagation(L,Clauses)) of
	false ->
	    case value_of(Rest) of
		false ->
		    false;
		Solution ->
		    [-L|Solution]
	    end;
	Solution ->
	    [L|Solution]
    end.

%% Speculation

speculate(F) ->
    Parent = self(),
    Pid = spawn_link(fun() -> Parent ! {self(),F()} end),
    {speculating,Pid}.

value_of({speculating,Pid}) ->
    receive {Pid,X} ->
	    X
    end.


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parallel 2
%% Worker pool 
%% 

start_pool(N) ->
    true = register(pool,spawn_link(fun()->pool([worker() || _ <- lists:seq(1,N)]) end)).

pool(Workers) ->
    pool(Workers,Workers).

pool(Workers,All) ->
    receive
	{get_worker,Pid} ->
	    case Workers of
		[] ->
		    Pid ! {pool,no_worker},
		    pool(Workers,All);
		[W|Ws] ->
		    Pid ! {pool,W},
		    pool(Ws,All)
	    end;
	{return_worker,W} ->
	    pool([W|Workers],All);
	{stop,Pid} ->
	    [unlink(W) || W <- All],
	    [exit(W,kill) || W <- All],
	    unregister(pool),
	    Pid ! {pool,stopped}
    end.

worker() ->
    spawn_link(fun work/0).

work() ->
    receive
	{task,Pid,R,F} ->
	    Pid ! {R,F()},
	    catch pool ! {return_worker,self()},
	    work()
    end.

speculate_on_worker(F) ->
    case whereis(pool) of
	undefined ->
	    ok; %% we're stopping
	Pool -> Pool ! {get_worker,self()}
    end,
    receive
	{pool,no_worker} ->
	    {not_speculating,F};
	{pool,W} ->
	    R = make_ref(),
	    W ! {task,self(),R,F},
	    {speculating,R}
    end.

worker_value_of({not_speculating,F}) ->
    F();
worker_value_of({speculating,R}) ->
    receive
	{R,X} ->
	    X
    end.


%% Solver speculating on worker pool. This goes a little bit faster
%% than the purely parallel speculative solution above, but for the
%% Sudoku solver, then using a worker pool is essential to getting
%% good performance... so it's important to show this.

pool_solve(P) ->
    start_pool(erlang:system_info(schedulers)-1),
    S = pool_solve1(P),
    pool ! {stop,self()},
    receive {pool,stopped} -> S end.

pool_solve1([]) ->
    [];
pool_solve1([[]|_]) ->
    false;
pool_solve1([[L|Lits]|Clauses]) ->
    Rest = speculate_on_worker(
	     fun()-> pool_solve1(unit_propagation(-L,[Lits|Clauses])) end),
    case pool_solve1(unit_propagation(L,Clauses)) of
	false ->
	    case worker_value_of(Rest) of
		false ->
		    false;
		Solution ->
		    [-L|Solution]
	    end;
	Solution ->
	    [L|Solution]
    end.


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parallel 3
%% Limit parallelism to large problems
limit_solve(P) ->
    start_pool(erlang:system_info(schedulers)-1),
    S = limit_solve1(P),
    pool ! {stop,self()},
    receive {pool,stopped} -> S end.

limit_solve1([]) ->
    [];
limit_solve1([[]|_]) ->
    false;
limit_solve1([[L|Lits]|Clauses]) when length(Clauses) < 250 ->
    solve([[L|Lits]|Clauses]);
limit_solve1([[L|Lits]|Clauses]) ->
    Rest = speculate_on_worker(
	     fun()-> limit_solve1(unit_propagation(-L,[Lits|Clauses])) end),
    case limit_solve1(unit_propagation(L,Clauses)) of
	false ->
	    case worker_value_of(Rest) of
		false ->
		    false;
		Solution ->
		    [-L|Solution]
	    end;
	Solution ->
	    [L|Solution]
    end.
