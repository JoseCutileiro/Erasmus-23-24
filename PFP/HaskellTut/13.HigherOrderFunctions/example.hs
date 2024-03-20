import Data.List
import System.IO

times5 :: Int -> Int
times5 x = x * 5

-- using our function (this is how map works)
applyOnList :: [Int] -> [Int]
applyOnList [] = []
applyOnList (x:xs) = times5 x : applyOnList xs

-- len 
len :: [Char] -> Int
len [] = 0
len (x:xs) = 1 + len xs

-- StrCompare using recursion
strcmp :: [Char] -> [Char] -> Bool
strcmp [] [] = True
strcmp (x:xs) (y:ys) = (len xs == len ys) && (x == y) && strcmp xs ys 

main = do

    -- Using map
    print (map times5 [1,2,3,4,5])

    -- Using our function
    print (applyOnList [1,2,3,4,5])

    --strcmp test
    print (strcmp "Jose33" "Jose3")