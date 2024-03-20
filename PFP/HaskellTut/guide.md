# Guide for haskell 

## Comments:

1. One-line 

```
-- This is a comment
```

2. Multi-line

```
{- 
MultiLine comment
MultiLine comment
-}
```

## Data types

(Types are static)

1. Int

```
Minimum: -2^63
Maximum: 2^63
```

2. Float

3. Double

4. Bool

5. Char

6. Tuple

## Math functions

1. sum: somar todos os números da lista

```
sum <list>

sumOfNums = sum [1..1000]
```

2. basic math

```
result = <num> <op> <num>

num :: [0-9]*
op :: + - * /
```

3. mod

```

result = mod <num> <num>

or

result = <num> 'mod' <num>
```

4. Note: sum with negative number

```
result = <num> + (<negative_num>)
```

5. More built in math: 

```
piVal = pi
ePow9 = exp 9
logOf9 = log 9
squared9 = 9 ** 2
truncateVal =  truncate 9.99
roundVal = round 9.99
ceilingVal = ceiling 9.99
floorVal = floor 9.99

(sin cos tan asin atan acos sinh tanh cosh asinh atanh acosh)
```

## Logical ops

```
result = <bool> <op> <bool>

ops -> && ||

result = not(<bool>)
```

## Lists

1. Concatenate
```
primeNumbers = [3,5,7,11]

morePrimes = primeNumbers ++ [13,17,19,23,29] 

-- Concatenate in the beggining

morePrimesV2 = 2 : morePrimes

```

2. Combine numbers into lists
```
nums = <num> : <num> : ... :[]
```

3. MultiList

```
multiList = [[3,5,7],[11,13,17]]
```
 
4. reverse list

```
result = reverse <list> 
```

5. Is empty? 

```
isListEmpty = null <List>
```

6. Get value of the list

```
firstValue = head <list>

secondValue = <list> !! 1

thirdValue = <list> !! 2

...
listValue = last <list>
```

7. Get Every value except the last

```
result = init <list>
```

8. Get the first N elements of the list

```

result = take <N> <list>

N: int
```

9. Remove the first N values of the list
```
result = drop <N> <list>

N: int
```

10. Check if value is in the list

```
result = <num> 'elem' <list>
```

11. Get maximum and minimum

```
resMax = maximum <list>
resMin = minimum <list>
```

12. Multiply every element of the list

```
result = product <list>
```

13. Cycle, replicate and repeat (see example 4)

## Modify elements in the list (list comprehension)

1. basic example with lists: see '5.UpdateLists'
2. list comprehension: see '6.ListComprehension'

## Tuples

1. basic examples with tuple: see '7.tupleBasic'

## Functions

1. See basic examples in "8.functions"
2. Declare a function: funcName :: arg1_type -> ... -> argn_type -> ret_type
3. make a function: funcName arg1 arg2 = operation (returned value)
4. Recursion example with haskell in "9.recursion"

## Guards

Instead of using if statements you can use guards  (very usefull) -> check example in "10.Guards"

This can be combine with the where clause, very usefull, you can see some example in "11.Where"

## Get list of items

See 12.ListItems to get more information 

## Higher order functions

You can see example in 13.HigherOrderFunctions

1. Map -> Aplicar uma função a todos os elementos da lista

```
result = map <func> <list>
```

## Functions inside functions

This is a bit weird and can make your code very confuse
so try to avoid this to be honest, but you can see example in 14.FunctionInsideFunctions

You can also return a function (see in the same example). Have the same problem of being confuse ...

## Labdas (anonymous functions)

This was very fast so is on 14.FunctionInsideFunctions too

## Other types

1. Enumerator types
2. Custom types
3. In custom types you can make complex types like

```
(remember this use uppercase)
(to see other example see 16.CustomTypes)

data Shape = Circle Float Float Float | Rectangle Float Float Float Float
    deriving Show

area :: Shape -> Float

area (Circle _ _ r) = pi * r ** 2
area (Rectangle x1 y1 x2 y2) = (abs (x2 - x1)) * (abs (y2 - y1))
```

## Type classes

This are the type classes: Num Eq Or Show

(generic classes)

Example: 
```hs
data Employee = Employee { name :: String,
                           position :: String,
                           idNum :: Int
                        } deriving (Eq, Show)
```

Here we are saying that we can see the string format of this, and we can also see if two of this intances are equal, this creae a default way of showing and seeing equality. But we can also define a custom

```hs

data ShirtSize S | M | L

instance Eq ShirtSize where
    S == S = True
    M == M = True
    L == L = True
    _ == _ = False

instance Show ShirtSize where
    show S = "Small"
    show M = "Medium"
    show L = "Large"

-- S in list
smallAvail = elem S [S, M, L]
```

## File IO

```
writeToFile = do 
    theFile <- openFile "file.txt" WriteMode
    hPutStrLn theFile ("Random text")
    hClose theFile

readFromFile = do 
    theFile <- openFile "file.txt" ReadMode
    content <- hGetContents theFile2
    putStr contents
    hClose theFile
```
