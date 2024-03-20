import Data.List
import System.IO

-- Recursive factorial
factorialRec :: Int -> Int
factorialRec 0 = 1
factorialRec n = n * factorialRec(n-1)

-- Non Recursive
factorial :: Int -> Int
factorial 0 = 1
factorial 1 = 1
factorial n = product [1..n]

input = 9 :: Int

main = do

    print (factorialRec input)
    print (factorial input)