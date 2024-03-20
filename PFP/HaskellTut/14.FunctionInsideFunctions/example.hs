import Data.List
import System.IO

-- Basic dupe function
dupe :: Int -> Int 
dupe x = x * 2

-- Receive a function and sets the argument for the function 3
doMult :: (Int -> Int) -> Int
doMult func = func 3

-- Returning a function
getAddFunc :: Int -> (Int -> Int)
getAddFunc x y = x + y

adds3 = getAddFunc 3
fourPlus3 = adds3 4

main = do

    print (doMult dupe)
    print (fourPlus3)

    -- This works with map too
    print (map adds3 [1,2,3,4])

    -- lambdas (anonymous functions)
    print (map (\x -> x * 2) [1..10])