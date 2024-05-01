-module(book).
-compile(export_all).

% This was writen with the help of the book 
% learnyousomeerlang.com

% Basic functio
addElements(A,B) -> A + B.

% list comprehensions
dupe(L) -> [2 * N || N <- L].


% .beam -> godgan/Bjorn Erlang abstract machine 

% pattern matching
% Note: io:format -> print

greet(male,Name) ->
        io:format("Hello, Mr. ~s!", [Name]);
greet(female,Name) ->
        io:format("Hello, Mrs. ~s!", [Name]);
greet(_,Name) ->
        io:format("Hello, ~s!", [Name]).

