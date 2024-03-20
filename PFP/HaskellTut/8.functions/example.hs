import Data.List
import System.IO


num7 = 7
getTriple x = x * 3

-- function type declaration
addMe :: Int -> Int -> Int
addTuples :: (Int, Int) -> (Int, Int) -> (Int, Int)

-- function logic defenition
addMe x y = x + y
addTuples (x1,y1) (x2,y2) = (x1 + x2, y1 + y2)


-- data with expected type (if you try other types this breaks)
x :: Int
y :: Int 
t1 :: (Int, Int)
t2 :: (Int, Int)

x = 10
y = 20
t1 = (3,6)
t2 = (6,9)

-- Different behaviour of the function according to input
whatAge :: Int -> String

whatAge 18 = "You can drive"
whatAge 2 = "GuguDada"
whatAge 21 = "You're an adult"
whatAge _ = "Insert generic response here"


-- similar to if
isOdd n
    | mod n 2 == 0 = False
    | otherwise = True

-- Main function
main = do

    -- Use function
    print (getTriple num7)

    -- Get information for the console
    putStrLn("What is your name?")
    name <- getLine
    putStrLn("Hello " ++ name)

    -- Use a functiom
    print (addMe x y)
    print (addTuples t1 t2)
    print (whatAge 2)

    print (isOdd 3)