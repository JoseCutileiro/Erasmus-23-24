# Guide: A full guide for haskell

This guide was based on two books
1. Real World haskell
2. Haskell: The craft of functional programming

# Haskell env

Three main components

1. ghc: compiler to generate native code
2. ghci: Interactive interpreter and debugger
3. runghc: Running haskell as scripts (you dont need to compile them)

# Operators (fast recap)

```
math: 
2 + 2 is equivalent to (+) 2 2
(the same for the remaining operators)

logic:
&& || == >= ...
(equal to C)

in lists:
	append
		[1,2,3] ++ [4,5] 
		[1,2,3,4,5]

		1 : [2,3]
		[1,2,3]

	head 
		head [1,2,3]
		1
	
	tail 
		tail [1,2,3]
		[2,3]
	
	take
		take 2 [1,2,3,4,5]
		[1,2]
	
	drop 
		[4,5]
	

special chars:
	\n -> newline
	\t -> tab

in strings:
	append
		"foo" ++ "bar"
		"foobar"

		'a' : "bc"
		"abc"
	
	(head, tail, take, drop)

```

# ghci

syntatic sugar in ghci -> keyword 'it'

it: uses the value from the previous interaction 

example
```
ghci> "foo"
"foo"
ghci> it ++ "bar"
"foobar"
```

# Higher order functions 

1. map 

```
map <func> <list>

map (2*) list

```

2. filter

```
filter <cond> <list>

example:

filter (<=10) list
```
















