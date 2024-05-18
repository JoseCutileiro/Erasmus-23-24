-module(test18).
-compile(export_all).

generate_random_list(Length) ->
    _ = rand:seed(exsplus, os:timestamp()),
    generate_random_list(Length, []).

generate_random_list(0, Acc) -> Acc;
generate_random_list(N, Acc) ->
    RandomElement = rand:uniform(),
    generate_random_list(N - 1, [RandomElement | Acc]).

merge_sort([]) -> [];
merge_sort([X]) -> [X];
merge_sort(Xs) -> 
    {Ys,Zs} = split(Xs),
    merge(merge_sort(Ys), merge_sort(Zs)).

merge([],Ys) -> Ys;
merge(Xs,[]) -> Xs;

merge([X|Xs],[Y|Ys]) when X =< Y -> 
    [X|merge(Xs,[Y|Ys])];

merge([X|Xs],[Y|Ys]) -> [Y|merge([X|Xs],Ys)].

split(Xs) -> split([],Xs,Xs).

split(L,[X|R],[_,_|Xs]) -> split([X|L],R,Xs);
split(L,R,_)           -> {L,R}.

merge_sort_par([]) -> [];
merge_sort_par([X]) -> [X];
merge_sort_par(Xs) -> 
    Parent = self(),
    {Ys,Zs} = split(Xs),
    L = length(Ys) + length(Zs),
    case L of 
        _ when L < 1000 -> merge(merge_sort(Ys),merge_sort(Zs));
        _ -> merge_spawn(Parent,Ys,Zs)
    end .

merge_spawn(Parent,Ys,Zs) ->
    Pid = spawn_link(
        fun() ->
            Parent ! {self(), merge_sort_par(Zs)}
        end),
    SortedYs = merge_sort_par(Ys),
    receive
        {Pid, SortedZs} -> merge(SortedYs, SortedZs)
    end.


